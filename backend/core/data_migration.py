"""Offline storage transaction shared by Electron and move_data_dir.py.

Only db/, chroma_projects/ and the metadata-repair journal move. A durable
intent journal permits rollback after a process interruption. No model imports.
"""
import argparse
import base64
from datetime import datetime, timezone
from itertools import chain
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import uuid

from core.storage_paths import resolve_data_dir, require_backend_offline, storage_locks

ITEMS = ('db', 'chroma_projects', 'metadata-repair-progress.sqlite',
         'metadata-repair-progress.sqlite-wal', 'metadata-repair-progress.sqlite-shm')
PENDING = 'data-dir-migration.json'
LAST = 'data-dir-last-migration.json'
MOVED = '.soundbot-data-moved.json'


def atomic_bytes(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows scanners can briefly hold a just-written journal.
                if os.name != 'nt' or attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8'))


def validate_roots(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Source and destination must be different, non-nested directories')
    if target.exists() and not target.is_dir():
        raise ValueError('Destination is not a directory')
    return source, target


def reject_links(root):
    root = Path(root)
    for path in chain([root], root.rglob('*')) if root.is_dir() else [root]:
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
            raise ValueError(f'Storage migration does not follow symlinks/junctions: {path}')


def inventory(root, names, *, hashes=True):
    result = {}
    for name in names:
        base = Path(root) / name
        if not base.exists():
            continue
        reject_links(base)
        paths = sorted(base.rglob('*')) if base.is_dir() else [base]
        for path in paths:
            relative = str(path.relative_to(root))
            if path.is_dir():
                result[relative] = {'directory': True}
                continue
            digest = None
            if hashes:
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            result[relative] = {'size': path.stat().st_size, 'sha256': digest}
    return result


def sqlite_summary(path):
    # No DatabaseManager/Chroma client: validation must not migrate schemas,
    # recover jobs, open models or create replacement collections.
    connection = sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True)
    try:
        if connection.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
            raise RuntimeError(f'SQLite quick_check failed: {path}')
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {name: connection.execute('SELECT COUNT(*) FROM "' + name.replace('"','""') + '"').fetchone()[0]
                for name in tables}
    finally:
        connection.close()


def validate_store(root):
    root = Path(root)
    main = root / 'db' / 'soundmind.db'
    if not main.is_file():
        raise ValueError(f'Destination/source has no SoundBot database: {main}')
    reject_links(root / 'db')
    summaries = {'db/soundmind.db': sqlite_summary(main)}
    if not {'files', 'file_artifacts', 'projects'}.issubset(summaries['db/soundmind.db']):
        raise ValueError('Not a compatible SoundBot library database')
    chroma = root / 'chroma_projects'
    if chroma.exists():
        reject_links(chroma)
        for path in chroma.rglob('chroma.sqlite3'):
            summaries[str(path.relative_to(root))] = sqlite_summary(path)
    progress = root / 'metadata-repair-progress.sqlite'
    if progress.exists():
        summaries[progress.name] = sqlite_summary(progress)
    return summaries


def _stage(record):
    return Path(record['target']) / ('.soundbot-move-' + record['id'])


def _save(user, record):
    write_json(Path(user) / PENDING, record)


def _remove_stage(record):
    stage = _stage(record)
    if not stage.exists():
        return
    # Recursive cleanup is restricted to our UUID-owned staging directory.
    if stage.resolve().parent != Path(record['target']).resolve():
        raise RuntimeError('Unsafe staging cleanup path')
    if not any(stage.iterdir()):
        stage.rmdir()
        return
    owner = json.loads((stage / '.owner.json').read_text(encoding='utf-8'))
    if owner != {'id': record['id']}:
        raise RuntimeError('Staging ownership mismatch; leaving files for inspection')
    reject_links(stage)
    shutil.rmtree(stage)


def _rollback(user, record, *, completed=False):
    source, target = validate_roots(record['source'], record['target'])
    # Keep startup blocked throughout rollback, including a second interruption.
    # Completed copies must remain retained when --recover resumes this work.
    completed = completed or record.get('rollback_completed', False)
    record['rollback_completed'] = completed
    _save(user, record)
    if record['mode'] == 'copy' and not completed and not _stage(record).exists():
        _stage(record).mkdir()
        write_json(_stage(record) / '.owner.json', {'id': record['id']})
    config = Path(user) / 'user_config.json'
    # Stop on conflicts; never replace a newly created source library.
    for name, archived in record['archives'].items():
        if name not in ITEMS or Path(archived).parent != source or not Path(archived).name.startswith(name + '.migrated-'):
            raise RuntimeError('Invalid archive path in migration journal')
        if Path(archived).exists() and (source / name).exists():
            raise RuntimeError(f'Rollback conflict at {source / name}; no data overwritten')
    for name, archived in record['archives'].items():
        if Path(archived).exists():
            Path(archived).rename(source / name)
    if record['config_before'] is None:
        if config.exists():
            config.unlink()
    else:
        atomic_bytes(config, base64.b64decode(record['config_before']))
    if record['mode'] == 'copy':
        for name in record['installed']:
            if name not in ITEMS:
                raise RuntimeError('Invalid installed item in migration journal')
            path = target / name
            if path.exists():
                if not completed and (_stage(record) / name).exists():
                    # Intent was journaled but rename never happened (e.g. a
                    # newly appeared target conflict). This is not our data.
                    continue
                if completed:
                    # The validation backend may have opened the copied library.
                    # Retain this copy as well when undoing a completed migration.
                    backup = target / (name + '.rolled-back-' + record['id'])
                    if backup.exists():
                        raise RuntimeError(f'Rollback destination already exists: {backup}')
                    path.rename(backup)
                else:
                    destination = _stage(record) / name
                    if destination.exists():
                        raise RuntimeError('Rollback staging conflict')
                    path.rename(destination)
    marker = source / MOVED
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8')).get('id') == record['id']:
        marker.unlink()
    _remove_stage(record)
    pending = Path(user) / PENDING
    if pending.exists():
        pending.unlink()
    record['phase'] = 'rolled_back'
    write_json(Path(user) / LAST, record)


def recover(user_data, *, transaction_id=None):
    require_backend_offline()
    user = Path(user_data).expanduser().resolve()
    path = user / (LAST if transaction_id else PENDING)
    record = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(record.get('id'), str) or not re_full_id(record['id']):
        raise ValueError('Invalid migration journal')
    if transaction_id and (record['id'] != transaction_id or record['phase'] != 'complete'):
        raise ValueError('Only the last completed migration can be rolled back')
    with storage_locks(user, record['source'], record['target']):
        require_backend_offline()
        if transaction_id and (user / PENDING).exists():
            raise RuntimeError('Interrupted migration found; run --recover before rolling back another transaction')
        if transaction_id and resolve_data_dir(user, {}) != Path(record['target']):
            raise RuntimeError('Configuration has changed since migration; refusing rollback')
        _rollback(user, record, completed=bool(transaction_id))
    return {'phase': 'rolled_back', 'id': record['id'], 'data_dir': record['source']}


def re_full_id(value):
    return len(value) == 32 and all(c in '0123456789abcdef' for c in value)


def migrate(user_data, destination, *, dry_run=False, use_existing=False, progress=None):
    # No storage stat/read/connection occurs until all backends are offline.
    require_backend_offline()
    user = Path(user_data).expanduser().resolve()
    source, target = validate_roots(resolve_data_dir(user), Path(destination).expanduser())
    if (user / PENDING).exists():
        raise RuntimeError('Interrupted migration found; run move_data_dir.py --recover first')
    if os.environ.get('SOUNDBOT_DATA_DIR') and not dry_run:
        raise ValueError('Unset SOUNDBOT_DATA_DIR before changing the persistent data directory')
    if not source.is_dir():
        raise ValueError('Source storage does not exist')
    existing = any((target / name).exists() for name in ITEMS)
    if (target / MOVED).exists():
        raise ValueError('Destination is an archived migration source; choose a new directory or roll back the last migration')
    if existing and not use_existing:
        raise FileExistsError('Destination already contains SoundBot data; explicitly use-existing or cancel')
    if use_existing and not existing:
        raise ValueError('Destination contains no existing library to use')
    # Reject mappings before resolve can conceal the lexical root itself.
    lexical_target = Path(destination).expanduser().absolute()
    for candidate in [lexical_target, *lexical_target.parents]:
        if candidate.is_symlink() or (hasattr(candidate, 'is_junction') and candidate.is_junction()):
            raise ValueError('A mapped/symlink destination is not supported')
    ancestor = target
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not os.access(ancestor, os.W_OK):
        raise PermissionError(f'Destination is not writable: {ancestor}')
    sizes = inventory(source, ITEMS, hashes=False)
    total = sum(value.get('size', 0) for value in sizes.values())
    required = 64 * 1024**2 if use_existing else total + max(64 * 1024**2, total // 10)
    free = shutil.disk_usage(ancestor).free
    if free < required:
        raise RuntimeError(f'Insufficient disk space: need {required} bytes, available {free}')
    plan = {'source': str(source), 'target': str(target), 'bytes': total,
            'required_bytes': required, 'free_bytes': free, 'mode': 'existing' if use_existing else 'copy'}
    if dry_run:
        return dict(plan, phase='dry_run')
    with storage_locks(user, source, target):
        # Recheck inside lifetime locks to close the GUI/startup race.
        require_backend_offline()
        if any((target / name).exists() for name in ITEMS) != existing:
            raise RuntimeError('Destination changed during preflight')
        config = user / 'user_config.json'
        before = config.read_bytes() if config.exists() else None
        settings = json.loads(before.decode('utf-8-sig')) if before else {}
        transaction = uuid.uuid4().hex
        record = dict(plan, id=transaction, phase='copying', installed=[], archives={},
                      config_before=base64.b64encode(before).decode() if before is not None else None)
        stage = _stage(record)
        stage.mkdir()
        emit = progress or (lambda event: None)
        try:
            write_json(stage / '.owner.json', {'id': transaction})
            _save(user, record)
            if use_existing:
                emit({'phase': 'validating_existing'})
                validate_store(target)
            else:
                emit({'phase': 'hashing_source'})
                manifest = inventory(source, ITEMS)
                source_counts = validate_store(source)
                for name in ITEMS:
                    path = source / name
                    if not path.exists():
                        continue
                    emit({'phase': 'copying', 'item': name})
                    if path.is_dir():
                        shutil.copytree(path, stage / name)
                    else:
                        shutil.copy2(path, stage / name)
                emit({'phase': 'verifying'})
                if inventory(stage, ITEMS) != manifest or inventory(source, ITEMS) != manifest:
                    raise RuntimeError('File manifest/size/SHA-256 validation failed or source changed')
                if validate_store(stage) != source_counts:
                    raise RuntimeError('SQLite table counts differ after copy')
                record['phase'] = 'installing'
                for name in ITEMS:
                    if (stage / name).exists():
                        if (target / name).exists():
                            raise FileExistsError(f'Destination conflict: {target / name}')
                        record['installed'].append(name)
                        _save(user, record)  # intent before rename for crash recovery
                        (stage / name).rename(target / name)
            record['phase'] = 'archiving'
            suffix = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + transaction[:8]
            for name in ITEMS:
                if (source / name).exists():
                    archived = source / (name + '.migrated-' + suffix)
                    if archived.exists():
                        raise FileExistsError(f'Archive conflict: {archived}')
                    record['archives'][name] = str(archived)
                    _save(user, record)
                    (source / name).rename(archived)
            write_json(source / MOVED, {'id': transaction, 'data_dir': str(target),
                'message': 'Data moved. Old folders are retained as *.migrated-*. Do not use an older SoundBot build.'})
            record['phase'] = 'configuring'
            _save(user, record)
            settings['dataDir'] = str(target)
            write_json(config, settings)
            record['phase'] = 'complete'
            _save(user, record)
            _remove_stage(record)
            os.replace(user / PENDING, user / LAST)
            emit({'phase': 'complete'})
            return dict(plan, phase='complete', id=transaction, archives=record['archives'])
        except BaseException:
            _rollback(user, record)
            raise


def cli(argv=None):
    parser = argparse.ArgumentParser(description='Move SoundBot databases offline with verification and rollback')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--to', type=Path)
    group.add_argument('--recover', action='store_true')
    group.add_argument('--rollback', metavar='TRANSACTION_ID')
    parser.add_argument('--user-data-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--use-existing', action='store_true')
    args = parser.parse_args(argv)
    if (args.recover or args.rollback) and (args.dry_run or args.use_existing):
        parser.error('--dry-run / --use-existing apply only to --to')
    try:
        if args.user_data_dir:
            user = args.user_data_dir
        else:
            import config
            user = config.get_user_data_dir()
        if args.recover or args.rollback:
            result = recover(user, transaction_id=args.rollback)
        else:
            result = migrate(user, args.to, dry_run=args.dry_run, use_existing=args.use_existing,
                             progress=lambda event: print(json.dumps(event), flush=True))
        print(json.dumps({'result': result}), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({'error': str(exc)}), flush=True)
        return 1
