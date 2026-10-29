"""Bounded automatic recovery. No SQLite transaction spans asynchronous work."""
from itertools import groupby, islice
import asyncio
import os
import threading
from contextlib import suppress
from weakref import WeakKeyDictionary
import time
import sqlite3
import logging

from core.index_publish import claim_scope, publish_batch

from core.audio_metadata import metadata_missing, probe_metadata
from core.audio_service import AudioServiceError

_resume_gates = WeakKeyDictionary()


async def resume_artifacts(db, project_id, job_id, kinds, targets, service, *,
                           run_audio, run_model, build_text, model_available,
                           shutting_down, batch_size=16):
    # Bound the total preparation footprint across projects as well as files.
    # Waiting projects remain cancellable; inference is shared by all projects.
    loop = asyncio.get_running_loop()
    gate = _resume_gates.setdefault(loop, asyncio.Lock())
    while True:
        job = db.get_job(job_id)
        if shutting_down() or (job and job['cancel_requested']):
            db.update_job(job_id, state='cancelled', stage='cancelled', processed=0)
            return dict.fromkeys(kinds, 0), True
        try:
            await asyncio.wait_for(gate.acquire(), timeout=.1)
            break
        except asyncio.TimeoutError:
            pass
    try:
        return await _resume_with_pool(db, project_id, job_id, kinds, targets, service,
            run_audio=run_audio, run_model=run_model, build_text=build_text,
            model_available=model_available, shutting_down=shutting_down, batch_size=batch_size)
    finally:
        gate.release()


async def _resume_with_pool(db, project_id, job_id, kinds, targets, service, *,
                           run_audio, run_model, build_text, model_available,
                           shutting_down, batch_size):
    from core.audio_service import AudioService
    from core.embedder import CLIPEmbedder, peek_embedder
    from core.index_audio import DecodePool, default_parallelism
    from core.indexer import AudioIndexer
    embedder = peek_embedder()
    if (isinstance(service, AudioService) and isinstance(embedder, CLIPEmbedder)
            and isinstance(targets.get('audio_vector'), AudioIndexer)
            and type(getattr(getattr(embedder, 'processor', None), 'feature_extractor', None)).__name__ == 'ClapFeatureExtractor'
            and model_available()
            and os.getenv('SOUNDBOT_DECODE_PROCESSES', str(default_parallelism())) != '0'
            and os.getenv('SOUNDBOT_FAST_MEL', '1') != '0'
            and os.getenv('SOUNDBOT_CONTINUOUS_INDEX', '1') != '0'):
        from core.index_pipeline import resume_continuous
        return await resume_continuous(db, project_id, job_id, kinds, targets, service,
            embedder=embedder, run_audio=run_audio, run_model=run_model,
            build_text=build_text, model_available=model_available, shutting_down=shutting_down)
    pool = None
    if (isinstance(service, AudioService) and isinstance(peek_embedder(), CLIPEmbedder)
            and isinstance(targets.get('audio_vector'), AudioIndexer)
            and os.getenv('SOUNDBOT_DECODE_PROCESSES', str(default_parallelism())) != '0'):
        pool = DecodePool(max(1, min(4, int(os.getenv('SOUNDBOT_DECODE_PROCESSES', str(default_parallelism()))))))
    try:
        return await _resume_artifacts(db, project_id, job_id, kinds, targets, service,
            run_audio=run_audio, run_model=run_model, build_text=build_text,
            model_available=model_available, shutting_down=shutting_down,
            batch_size=batch_size, decode_pool=pool)
    finally:
        if pool is not None:
            pool.cancel.set()
            await run_audio(pool.close)


async def _resume_artifacts(db, project_id, job_id, kinds, targets, service, *,
                           run_audio, run_model, build_text, model_available,
                           shutting_down, batch_size=16, decode_pool=None):
    failures = dict.fromkeys(kinds, 0)
    processed = 0
    cancelled = False
    last_check = 0.0

    def cancellation_requested(force=False):
        nonlocal cancelled, last_check
        if shutting_down():
            cancelled = True
        now = time.monotonic()
        if force or now - last_check >= 0.5:
            job = db.get_job(job_id)
            cancelled = cancelled or bool(job and job['cancel_requested'])
            last_check = now
        return cancelled

    groups = groupby(db.iter_artifacts_for_work(project_id, kinds),
                     key=lambda row: row['file_uuid'])
    while True:
        # Materialize each group immediately; groupby shares its iterator.
        batch = [list(rows) for _, rows in islice(groups, max(1, batch_size))]
        if not batch or cancellation_requested(force=True):
            break
        claimed = []
        with claim_scope(db, project_id, claimed):
            skipped = 0
            with db.transaction():
                for rows in batch:
                    for row in rows:
                        if db.claim_artifact_for_work(project_id, row['file_uuid'], row['kind']):
                            claimed.append(row)
                        else:
                            skipped += 1

            outcomes = []
            metadata_updates = {}
            probes = {}
            waves = []
            audio = []
            texts = []

            def outcome(row, state='ready', **kwargs):
                outcomes.append((row, state, kwargs))
                if state in ('failed', 'pending'):
                    failures[row['kind']] += 1

            def failure(row, exc):
                outcome(row, 'failed',
                        error_code=exc.code if isinstance(exc, AudioServiceError) else f"{row['kind']}_failed",
                        error_message=exc.message if isinstance(exc, AudioServiceError) else str(exc))

            from core.audio_service import AudioService
            from core.embedder import CLIPEmbedder, peek_embedder
            from core.index_audio import prepare_index_audio, check_cancel, IndexWorkCancelled, default_parallelism
            from core.indexer import AudioIndexer

            stop = decode_pool.cancel if decode_pool else threading.Event()
            slots = asyncio.Semaphore(max(1, min(8, int(os.getenv('SOUNDBOT_INDEX_PREFETCH', str(default_parallelism()))))))
            embedder = peek_embedder()
            can_fuse = (isinstance(service, AudioService) and isinstance(embedder, CLIPEmbedder)
                        and isinstance(targets.get('audio_vector'), AudioIndexer))

            async def prepare_file(rows):
                async with slots:
                    prepared_audio = None
                    attempted_decode = False
                    wants_audio = any(row['kind'] == 'audio_vector' for row in rows)
                    wants_wave = any(row['kind'] == 'waveform' for row in rows)
                    try:
                        for row in rows:
                            if cancellation_requested() or stop.is_set():
                                stop.set()
                                break
                            file_id, kind = row['file_uuid'], row['kind']
                            try:
                                if metadata_missing(row):
                                    if file_id not in probes:
                                        try:
                                            probes[file_id] = await run_audio(probe_metadata, service, row['path'])
                                        except Exception as exc:
                                            probes[file_id] = exc
                                    metadata = probes[file_id]
                                    if isinstance(metadata, Exception):
                                        raise metadata
                                    metadata_updates[file_id] = metadata
                                    row.update(metadata.to_dict())
                                check_cancel(stop)
                                if (can_fuse and wants_audio and model_available()
                                        and kind in ('audio_vector', 'waveform') and not attempted_decode):
                                    attempted_decode = True
                                    decode = decode_pool.prepare if decode_pool else prepare_index_audio
                                    prepared_audio = await run_audio(decode, service,
                                        row['path'], embedder.sample_rate, waveform=wants_wave, cancel=stop)
                                check_cancel(stop)
                                if kind == 'waveform':
                                    if prepared_audio is not None:
                                        waveform = prepared_audio.waveform
                                    elif isinstance(service, AudioService):
                                        waveform = await run_audio(service.waveform, row['path'], cancel=stop)
                                    else:
                                        waveform = await run_audio(service.waveform, row['path'])
                                    check_cancel(stop)
                                    waves.append((row, waveform))
                                    outcome(row)
                                elif kind == 'audio_vector':
                                    if not model_available():
                                        outcome(row, 'pending', error_code='model_unavailable',
                                                error_message='CLAP model is not installed or ready')
                                        continue
                                    metadata = {key: row[key] for key in
                                                ('filename', 'duration', 'sample_rate', 'channels', 'tags')}
                                    metadata.update(file_id=file_id, size=row['file_size'], project_id=project_id)
                                    options = {'prepared_audio': prepared_audio, 'cancel': stop} if prepared_audio else {}
                                    prepared = await run_model(targets[kind].prepare_audio, row['path'], metadata, **options)
                                    check_cancel(stop)
                                    if prepared is None:
                                        raise RuntimeError('CLAP indexer returned no vector')
                                    audio.append((row, prepared))
                                else:
                                    prepared = await targets[kind].prepare_metadata([build_text(row)])
                                    check_cancel(stop)
                                    if len(prepared) != 1:
                                        raise RuntimeError('metadata embedding returned no vector')
                                    texts.append((row, prepared[0]))
                            except IndexWorkCancelled:
                                break
                            except Exception as exc:
                                if stop.is_set():
                                    break
                                failure(row, exc)
                    finally:
                        if prepared_audio is not None:
                            prepared_audio.close()

            async def watch_cancel():
                while True:
                    await asyncio.sleep(0.1)
                    if cancellation_requested(force=True):
                        stop.set()
                        return

            children = [asyncio.create_task(prepare_file(list(rows)))
                        for _, rows in groupby(claimed, key=lambda row: row['file_uuid'])]
            watcher = asyncio.create_task(watch_cancel())
            joined = asyncio.gather(*children)
            try:
                # Shield the actual work: cancelling an executor await does not
                # stop its thread. Drain workers before closing spools/restoring
                # claims, so no late write can race a resumed job.
                await asyncio.shield(joined)
            except BaseException:
                stop.set()
                await asyncio.gather(*children, return_exceptions=True)
                raise
            finally:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher

            # Write Chroma first, then publish ready states in SQLite. A crash in
            # between leaves processing, recovered to pending on the next startup.
            async def durable_write(method, rows):
                task = asyncio.create_task(run_model(method, rows))
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    # A thread doing an upsert cannot be cancelled. Wait before
                    # releasing claims so a restarted pass cannot race it.
                    await asyncio.gather(task, return_exceptions=True)
                    raise

            async def write_vectors(kind, rows, method):
                if not rows:
                    return
                try:
                    await durable_write(method, [prepared for _, prepared in rows])
                    for row, prepared in rows:
                        outcome(row, engine_fingerprint=prepared['metadata']['engine_fingerprint'])
                except Exception:
                    # Idempotent upserts permit isolation after a partial batch
                    # failure; never recompute inference or fail unrelated rows.
                    for row, prepared in rows:
                        try:
                            await durable_write(method, [prepared])
                            outcome(row, engine_fingerprint=prepared['metadata']['engine_fingerprint'])
                        except Exception as exc:
                            failure(row, exc)
            if audio:
                await write_vectors('audio_vector', audio, targets['audio_vector'].write_prepared_audio)
            if texts:
                await write_vectors('text_vector', texts, targets['text_vector'].write_prepared_metadata)

            published = publish_batch(db, project_id, outcomes, metadata_updates, waves)
            prior_states = {(row['file_uuid'], row['kind']): state for row, state, _ in outcomes}
            for row, state, _ in published:
                if state == 'failed' and prior_states[(row['file_uuid'], row['kind'])] not in ('failed', 'pending'):
                    failures[row['kind']] += 1
            finished = {(row['file_uuid'], row['kind']) for row, _, _ in published}
            # The scope restores only claims not successfully published, including
            # cancellation and fatal exceptions. No cleanup writes on the happy path.
            claimed[:] = [row for row in claimed if (row['file_uuid'], row['kind']) not in finished]
            processed += len(published) + skipped
            for attempt in range(2):
                try:
                    db.update_job(job_id, processed=processed, stage='processing')
                    break
                except sqlite3.Error:
                    logging.getLogger(__name__).exception('Progress update failed (attempt %s/2)', attempt + 1)
            if cancelled:
                break
    if cancelled:
        db.update_job(job_id, state='cancelled', stage='cancelled', processed=processed)
    else:
        # A transient failure of the last periodic progress write must not leave
        # a completed job showing an older count.
        db.update_job(job_id, processed=processed)
    return failures, cancelled
