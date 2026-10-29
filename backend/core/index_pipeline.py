"""Continuous, bounded prepare -> cross-file inference -> durable publication.

Used only for the real local CLAP pipeline. Generic/external adapters retain the
recovery path in index_resume. SQLite claims stay on the event-loop thread;
publication uses a separate connection/worker and never overlaps a transaction
with asynchronous work.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
import logging
import os
import sqlite3

import numpy as np
import torch

from core.audio_metadata import metadata_missing
from core.audio_service import AudioServiceError
from core.embedder import normalize_embedding
from core.index_audio import IndexWorkCancelled, default_parallelism
from core.index_features import FeatureSpec
from core.index_publish import claim_scope, publish_batch
from core.index_workers import FeaturePool

logger = logging.getLogger(__name__)


async def resume_continuous(db, project_id, job_id, kinds, targets, service, *,
                            embedder, run_audio, run_model, build_text,
                            model_available, shutting_down, pool_factory=FeaturePool):
    workers = max(1, min(4, int(os.getenv('SOUNDBOT_DECODE_PROCESSES', str(default_parallelism())))))
    limit = max(workers, min(64, int(os.getenv('SOUNDBOT_PIPELINE_FILES', '64'))))
    batch_size = max(1, min(16, int(os.getenv('SOUNDBOT_CLAP_BATCH_SIZE', '16'))))
    pool = pool_factory(workers, FeatureSpec.from_embedder(embedder))
    stop = pool.cancel
    failures = dict.fromkeys(kinds, 0)
    states = {}
    claimed = []
    slots = asyncio.Semaphore(limit)
    completed = asyncio.Queue(maxsize=limit)
    features_done = asyncio.Event()
    abort = asyncio.Event()
    file_tasks = set()
    processed = skipped = 0
    cancelled = False
    watch_errors = []
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='index-publish')
    loop = asyncio.get_running_loop()

    def stopping():
        nonlocal cancelled
        job = db.get_job(job_id)
        if shutting_down() or (job and job['cancel_requested']):
            cancelled = True
            stop.set()
        return stop.is_set()

    def wake_files():
        for state in states.values():
            state['ready'].set()

    async def watch():
        try:
            while True:
                await asyncio.sleep(.05)
                if stopping():
                    wake_files()
                    return
        except Exception as exc:
            watch_errors.append(exc)
            abort.set(); stop.set(); wake_files()

    def fail(row, exc):
        return (row, 'failed', dict(
            error_code=exc.code if isinstance(exc, AudioServiceError) else f"{row['kind']}_failed",
            error_message=exc.message if isinstance(exc, AudioServiceError) else str(exc)))

    async def infer(entries):
        if not entries or stop.is_set():
            return
        inputs = {'input_features': torch.from_numpy(np.concatenate([e[2] for e in entries])),
                  'is_longer': torch.zeros((len(entries), 1), dtype=torch.bool)}
        try:
            outputs = await run_model(embedder._forward_audio_batch, inputs, stop)
            if len(outputs) != len(entries):
                raise RuntimeError('Audio feature batch returned wrong result count')
        except IndexWorkCancelled:
            return
        except Exception as exc:
            # Model failures, like malformed features, must be attributed to
            # their file, not to every neighbour sharing the GPU batch.
            if len(entries) > 1:
                middle = len(entries) // 2
                await infer(entries[:middle]); await infer(entries[middle:])
                return
            state = states[entries[0][0]]
            state['error'] = exc
            outputs = [None]
        for (file_id, sequence, _), vector in zip(entries, outputs):
            state = states[file_id]
            if sequence != state['count']:
                state['error'] = RuntimeError('Out-of-order audio window')
            if vector is not None and state['error'] is None:
                value = normalize_embedding(vector).astype(np.float64)
                state['total'] = value if state['total'] is None else state['total'] + value
            state['count'] += 1
            if state['count'] == state['expected']:
                state['ready'].set()

    async def consume():
        try:
            while not features_done.is_set() and not stop.is_set():
                entries = await asyncio.to_thread(pool.read_batch, batch_size)
                await infer(entries)
        except BaseException:
            abort.set(); stop.set(); wake_files()
            raise

    async def prepare_file(file_id, rows):
        state = states[file_id]
        result = dict(rows=rows, outcomes=[], metadata=None, waves=[], audio=[], text=None)
        try:
            wants_audio = any(r['kind'] == 'audio_vector' for r in rows)
            wants_wave = any(r['kind'] == 'waveform' for r in rows)
            need_metadata = any(metadata_missing(r) for r in rows)
            stream = await asyncio.wrap_future(pool.submit(file_id, rows[0]['path'], service,
                embedder.sample_rate, audio=wants_audio, waveform=wants_wave, need_metadata=need_metadata))
            if hasattr(pool, 'stats'):
                for key in pool.stats:
                    pool.stats[key] += stream[key]
            state['expected'] = stream['windows']
            if state['count'] == state['expected']:
                state['ready'].set()
            await state['ready'].wait()
            if stop.is_set() or stream['cancelled']:
                return
            metadata = stream['metadata']
            if metadata is not None:
                result['metadata'] = metadata
                for row in rows:
                    row.update(metadata.to_dict())
            for row in rows:
                kind = row['kind']
                try:
                    if need_metadata and metadata is None:
                        raise stream['decode_error'] or ValueError('No metadata returned')
                    if kind == 'waveform':
                        if stream['waveform'] is None:
                            raise stream['decode_error'] or ValueError('No waveform returned')
                        result['waves'].append((row, stream['waveform']))
                        result['outcomes'].append((row, 'ready', {}))
                    elif kind == 'audio_vector':
                        error = stream['decode_error'] or stream['feature_error'] or state['error']
                        if error is not None:
                            raise error
                        if not state['count']:
                            raise ValueError('音频为空')
                        vector = normalize_embedding(state['total'] / state['count'])
                        meta = {key: row[key] for key in ('filename', 'duration', 'sample_rate', 'channels', 'tags')}
                        meta.update(file_id=file_id, size=row['file_size'], project_id=project_id)
                        prepared = await run_audio(targets[kind].prepare_audio, row['path'], meta,
                                                   embedding=vector, cancel=stop)
                        if prepared is None:
                            raise RuntimeError('CLAP indexer returned no vector')
                        result['audio'].append((row, prepared))
                    else:
                        result['text'] = (row, await run_audio(build_text, row))
                except IndexWorkCancelled:
                    return
                except Exception as exc:
                    result['outcomes'].append(fail(row, exc))
            if not stop.is_set():
                await completed.put(result)
        except IndexWorkCancelled:
            return
        except Exception as exc:
            # Worker transport failures are real job failures: partial feature
            # messages cannot safely be matched to a silently replaced worker.
            abort.set(); stop.set(); wake_files()
            raise

    async def schedule():
        nonlocal skipped
        tasks = []
        try:
            groups = groupby(db.iter_artifacts_for_work(project_id, kinds), key=lambda r: r['file_uuid'])
            exhausted = False
            while not exhausted and not stop.is_set():
                while not stop.is_set():
                    try:
                        await asyncio.wait_for(slots.acquire(), timeout=.05)
                        break
                    except asyncio.TimeoutError:
                        continue
                if stopping():
                    break
                page = []
                # Amortize claim transactions without adding a completion
                # barrier. Slots cover every claimed/unpublished file globally.
                for _ in range(8):
                    try:
                        file_id, grouped = next(groups)
                    except StopIteration:
                        slots.release(); exhausted = True
                        break
                    page.append((file_id, list(grouped)))
                    if len(page) == 8 or slots.locked():
                        break
                    await slots.acquire()
                selected_page = []
                with db.transaction():
                    for file_id, rows in page:
                        selected = []
                        for row in rows:
                            if db.claim_artifact_for_work(project_id, file_id, row['kind']):
                                selected.append(row); claimed.append(row)
                            else:
                                skipped += 1
                        selected_page.append((file_id, selected))
                for file_id, selected in selected_page:
                    if not selected:
                        slots.release()
                        continue
                    states[file_id] = dict(total=None, count=0, expected=None, error=None, ready=asyncio.Event())
                    task = asyncio.create_task(prepare_file(file_id, selected))
                    file_tasks.add(task); tasks.append(task)
                for task in [t for t in tasks if t.done()]:
                    task.result(); file_tasks.discard(task); tasks.remove(task)
                await asyncio.sleep(0)
            if tasks:
                await asyncio.gather(*tasks)
        finally:
            features_done.set()
            await completed.put(None)

    async def text_rows(pairs):
        if not pairs or stop.is_set():
            return [], []
        try:
            prepared = await targets['text_vector'].prepare_metadata([payload for _, payload in pairs])
            byid = {item['file_id']: item for item in prepared}
            if set(byid) != {row['file_uuid'] for row, _ in pairs}:
                raise RuntimeError('Text batch returned mismatched file identities')
            return [(row, byid[row['file_uuid']]) for row, _ in pairs], []
        except Exception as exc:
            if len(pairs) == 1:
                return [], [fail(pairs[0][0], exc)]
            middle = len(pairs) // 2
            left, le = await text_rows(pairs[:middle]); right, re = await text_rows(pairs[middle:])
            return left + right, le + re

    def publish(items, texts, text_errors):
        outcomes = [o for item in items for o in item['outcomes']] + text_errors
        for kind, rows in (('audio_vector', [pair for item in items for pair in item['audio']]),
                           ('text_vector', texts)):
            if stop.is_set():
                return []
            if not rows:
                continue
            method = (targets[kind].write_prepared_audio if kind == 'audio_vector'
                      else targets[kind].write_prepared_metadata)
            try:
                method([prepared for _, prepared in rows])
                outcomes.extend((row, 'ready', {'engine_fingerprint': prepared['metadata']['engine_fingerprint']})
                                for row, prepared in rows)
            except Exception:
                for row, prepared in rows:
                    if stop.is_set():
                        return []
                    try:
                        method([prepared])
                        outcomes.append((row, 'ready', {'engine_fingerprint': prepared['metadata']['engine_fingerprint']}))
                    except Exception as exc:
                        outcomes.append(fail(row, exc))
        metadata = {item['rows'][0]['file_uuid']: item['metadata'] for item in items if item['metadata'] is not None}
        if stop.is_set():
            return []
        return publish_batch(db, project_id, outcomes, metadata,
                             [wave for item in items for wave in item['waves']])

    async def publishing():
        nonlocal processed
        try:
            ended = False
            while not ended:
                items = []
                item = await completed.get()
                if item is None:
                    return
                items.append(item)
                deadline = loop.time() + .5
                while len(items) < 32:
                    try:
                        item = await asyncio.wait_for(completed.get(), max(.001, deadline - loop.time()))
                    except asyncio.TimeoutError:
                        break
                    if item is None:
                        ended = True
                        break
                    items.append(item)
                    if loop.time() >= deadline:
                        break
                if abort.is_set():
                    continue
                texts, errors = await text_rows([item['text'] for item in items if item['text'] is not None])
                if stop.is_set():
                    continue
                # Future is shielded; parent cancellation cannot abandon a
                # Chroma write before claim_scope restores the same rows.
                future = loop.run_in_executor(writer, publish, items, texts, errors)
                published = await asyncio.shield(future)
                finished = {(row['file_uuid'], row['kind']) for row, _, _ in published}
                claimed[:] = [r for r in claimed if (r['file_uuid'], r['kind']) not in finished]
                for row, state, _ in published:
                    failures[row['kind']] += int(state in ('failed', 'pending'))
                processed += len(published)
                for item in items:
                    states.pop(item['rows'][0]['file_uuid'], None)
                    slots.release()
                for attempt in range(2):
                    try:
                        db.update_job(job_id, processed=processed + skipped, stage='processing')
                        break
                    except sqlite3.Error:
                        logger.exception('Progress update failed (attempt %s/2)', attempt + 1)
        except BaseException:
            abort.set(); stop.set(); wake_files()
            # Unblock a producer's final sentinel even after publication fails.
            while not completed.empty():
                completed.get_nowait()
            raise

    watcher = asyncio.create_task(watch(), name='index-cancel-watch')
    with claim_scope(db, project_id, claimed):
        tasks = [asyncio.create_task(fn()) for fn in (schedule, consume, publishing)]
        joined = asyncio.gather(*tasks)
        try:
            await asyncio.shield(joined)
            if watch_errors:
                raise watch_errors[0]
        except BaseException:
            abort.set(); stop.set(); wake_files()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(*file_tasks, return_exceptions=True)
            raise
        finally:
            stop.set(); wake_files()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            # Queue teardown and pending writer completion precede claim repair.
            def close_writer_connection():
                connection = getattr(db._local, 'conn', None)
                if connection is not None:
                    connection.close()
                    db._local.conn = None
            try:
                await asyncio.to_thread(pool.close)
            finally:
                try:
                    await loop.run_in_executor(writer, close_writer_connection)
                finally:
                    await asyncio.to_thread(writer.shutdown, wait=True)
    db.update_job(job_id, processed=processed + skipped,
                  **({'state': 'cancelled', 'stage': 'cancelled'} if cancelled else {}))
    return failures, cancelled
