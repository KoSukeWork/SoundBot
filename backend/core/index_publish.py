"""Short, fault-isolated SQLite publication after durable Chroma writes."""
from contextlib import contextmanager
import logging
import sqlite3

logger = logging.getLogger(__name__)


def _restore_claim(db, project_id, row):
    # A committed ready/failed result, or a concurrent invalidation, must survive.
    with db.get_cursor() as cursor:
        cursor.execute('''UPDATE file_artifacts SET state=?,error_code=?,error_message=?
            WHERE project_id=? AND file_uuid=? AND kind=? AND state='processing' ''',
            (row['state'], row.get('error_code'), row.get('error_message'),
             project_id, row['file_uuid'], row['kind']))


@contextmanager
def claim_scope(db, project_id, claimed):
    """No transaction is held here. Covers exceptions and asyncio cancellation."""
    try:
        yield
    finally:
        if claimed:
            restored = False
            for _ in range(2):
                try:
                    with db.transaction():
                        for row in claimed:
                            _restore_claim(db, project_id, row)
                    restored = True
                    break
                except sqlite3.Error:
                    logger.exception('Batch claim restoration failed; retrying or isolating rows')
            if not restored:
                errors = []
                for row in claimed:
                    try:
                        with db.transaction():
                            _restore_claim(db, project_id, row)
                    except sqlite3.Error as exc:
                        errors.append(f"{row['file_uuid']}/{row['kind']}: {exc}")
                if errors:
                    raise RuntimeError('Database remains unwritable; claims could not be restored: '
                                       + '; '.join(errors))


def _failed(row, exc):
    return (row, 'failed', {'error_code': f"{row['kind']}_persist_failed",
                            'error_message': str(exc)})


def _publish_one(db, project_id, item, waveforms):
    row, state, kwargs = item
    if row['kind'] == 'waveform' and state == 'ready':
        waveform = waveforms[row['file_uuid']]
        try:
            # update_peaks can execute several statements. Roll all of those
            # back for this item before publishing its failure state.
            with db.transaction():
                saved = db.update_peaks(row['path'], waveform.peaks, project_id,
                    source_fingerprint=waveform.fingerprint.source_key,
                    waveform_fingerprint=waveform.fingerprint.key,
                    waveform_version=int(waveform.fingerprint.waveform_version),
                    raise_errors=True)
                if not saved:
                    raise ValueError('Waveform save matched no file (canonical_path/project mismatch or file removed)')
            return item
        except sqlite3.Error:
            raise  # Whole-batch retry first, then per-artifact fallback.
        except Exception as exc:
            item = _failed(row, exc)
            _, state, kwargs = item
    db.set_artifact_state(project_id, row['file_uuid'], row['kind'], state,
                          source_fingerprint=row['source_fingerprint'], **kwargs)
    return item


def publish_batch(db, project_id, outcomes, metadata_updates, waves):
    # Waveform invalidation must happen before vector ready states, including
    # in the per-artifact fallback path.
    ordered = sorted(outcomes, key=lambda item: item[0]['kind'] != 'waveform')
    waveforms = {row['file_uuid']: waveform for row, waveform in waves}
    for attempt in range(2):
        try:
            with db.transaction():
                for file_id, metadata in metadata_updates.items():
                    db.update_audio_metadata(project_id, file_id, metadata)
                published = [_publish_one(db, project_id, item, waveforms) for item in ordered]
            return published
        except sqlite3.Error:
            logger.exception('SQLite batch publication failed (attempt %s/2)', attempt + 1)

    published, errors = [], []
    for item in ordered:
        row = item[0]
        try:
            with db.transaction():
                metadata = metadata_updates.get(row['file_uuid'])
                if metadata:
                    db.update_audio_metadata(project_id, row['file_uuid'], metadata)
                result = _publish_one(db, project_id, item, waveforms)
            published.append(result)
        except sqlite3.Error as exc:
            # The row's transaction was rolled back. Persist just its error,
            # without rerunning inference or preventing other rows from publishing.
            failed = _failed(row, exc)
            try:
                with db.transaction():
                    _publish_one(db, project_id, failed, waveforms)
                published.append(failed)
            except sqlite3.Error as mark_error:
                errors.append(f"{row['file_uuid']}/{row['kind']}: {mark_error}")
    if errors:
        raise RuntimeError('Could not persist artifact errors; database remains unwritable: '
                           + '; '.join(errors))
    return published
