"""Bounded automatic recovery. No SQLite transaction spans asynchronous work."""
from itertools import groupby, islice
import time
import sqlite3
import logging

from core.index_publish import claim_scope, publish_batch

from core.audio_metadata import metadata_missing, probe_metadata
from core.audio_service import AudioServiceError


async def resume_artifacts(db, project_id, job_id, kinds, targets, service, *,
                           run_audio, run_model, build_text, model_available,
                           shutting_down, batch_size=16):
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

            for row in claimed:
                if cancellation_requested():
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
                    if kind == 'waveform':
                        waveform = await run_audio(service.waveform, row['path'])
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
                        prepared = await run_model(targets[kind].prepare_audio, row['path'], metadata)
                        if prepared is None:
                            raise RuntimeError('CLAP indexer returned no vector')
                        audio.append((row, prepared))
                    else:
                        prepared = await targets[kind].prepare_metadata([build_text(row)])
                        if len(prepared) != 1:
                            raise RuntimeError('metadata embedding returned no vector')
                        texts.append((row, prepared[0]))
                except Exception as exc:
                    failure(row, exc)

            # Write Chroma first, then publish ready states in SQLite. A crash in
            # between leaves processing, recovered to pending on the next startup.
            if audio:
                try:
                    await run_model(targets['audio_vector'].write_prepared_audio,
                                    [prepared for _, prepared in audio])
                    for row, prepared in audio:
                        outcome(row, engine_fingerprint=prepared['metadata']['engine_fingerprint'])
                except Exception as exc:
                    for row, _ in audio:
                        failure(row, exc)
            if texts:
                try:
                    await run_model(targets['text_vector'].write_prepared_metadata,
                                    [prepared for _, prepared in texts])
                    for row, prepared in texts:
                        outcome(row, engine_fingerprint=prepared['metadata']['engine_fingerprint'])
                except Exception as exc:
                    for row, _ in texts:
                        failure(row, exc)

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
