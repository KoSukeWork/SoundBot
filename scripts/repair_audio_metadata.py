"""Offline, resumable metadata repair. Never computes or replaces embeddings.

Close SoundBot first. Without --apply this only reports SQLite candidates.
An interrupted run repeats its last page safely. Failed files are journaled
and retried once on the next invocation; successful repairs are idempotent.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))


def require_offline():
    """Conservative process check before opening either application store."""
    if os.name == 'nt':
        result = subprocess.run([
            'powershell', '-NoProfile', '-NonInteractive', '-Command',
            "Get-CimInstance Win32_Process | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
        ], capture_output=True, text=True, check=True)
        processes = json.loads(result.stdout)
        processes = processes if isinstance(processes, list) else [processes]
        for process in processes:
            command = (process.get('CommandLine') or '').lower().replace('\\', '/')
            name = (process.get('Name') or '').lower()
            if ('soundbot-backend' in name or name == 'soundbot.exe'
                    or ('python' in name and ('backend/main.py' in command
                        or ('soundbot' in command and 'main.py' in command)))):
                raise RuntimeError(f"SoundBot is still running (PID {process['ProcessId']}); close it normally first")
    else:
        result = subprocess.run(['ps', '-eo', 'args='], capture_output=True, text=True, check=True)
        if any('soundbot-backend' in line.lower() or 'backend/main.py' in line
               for line in result.stdout.splitlines()):
            raise RuntimeError('SoundBot is still running; close it normally first')


def repair_page(conn, rows, collections, service):
    """Update existing vectors, then SQLite. Retry is safe after either write.

    Chroma update merges the three supplied fields. IDs, embeddings, documents,
    engine/source fingerprints, other metadata and artifact states are preserved.
    """
    from core.audio_metadata import metadata_missing, probe_metadata
    updates, failed = {}, {}
    for row in rows:
        try:
            values = {key: row[key] for key in ('duration', 'sample_rate', 'channels')}
            if metadata_missing(values):
                probed = probe_metadata(service, row['path'])
                values = {key: getattr(probed, key) for key in values}
            updates[row['file_uuid']] = values
        except Exception as exc:
            failed[row['id']] = str(exc)
    changed_vectors = 0
    for kind, collection in collections.items():
        eligible = [row for row in rows if row['file_uuid'] in updates and row[kind]]
        if not eligible:
            continue
        if collection is None:
            raise RuntimeError(f'Active {kind} collection is missing; no replacement collection was created')
        snapshot = collection.get(ids=[row['file_uuid'] for row in eligible], include=['metadatas'])
        existing = dict(zip(snapshot['ids'], snapshot['metadatas']))
        ids, metadata = [], []
        for row in eligible:
            file_id = row['file_uuid']
            current = existing.get(file_id)
            if current is None:
                # A processing row may never have written a vector. A ready
                # row with no vector is corruption requiring explicit reconcile.
                if row[kind] == 'ready':
                    failed[row['id']] = f'{kind}: ready vector is missing; run explicit reconcile after repair'
                continue
            desired = updates[file_id]
            if any(current.get(key) != value for key, value in desired.items()):
                ids.append(file_id)
                metadata.append(desired)
        if ids:
            collection.update(ids=ids, metadatas=metadata)
            changed_vectors += len(ids)
    changed_files = 0
    with conn:
        for row in rows:
            desired = updates.get(row['file_uuid'])
            if desired and any(row[key] != value for key, value in desired.items()):
                conn.execute('''UPDATE files SET duration=?,sample_rate=?,channels=?,updated_at=CURRENT_TIMESTAMP
                                WHERE id=?''', (desired['duration'], desired['sample_rate'],
                                              desired['channels'], row['id']))
                changed_files += 1
    return {'files_updated': changed_files, 'vectors_updated': changed_vectors, 'failures': failed}


SELECT_ROWS = '''SELECT f.id,f.file_uuid,f.path,f.project_id,f.duration,f.sample_rate,f.channels,
    av.state AS audio_vector, tv.state AS text_vector
    FROM files f
    LEFT JOIN file_artifacts av ON av.project_id=f.project_id AND av.file_uuid=f.file_uuid
         AND av.kind='audio_vector' AND av.state IN ('ready','processing')
    LEFT JOIN file_artifacts tv ON tv.project_id=f.project_id AND tv.file_uuid=f.file_uuid
         AND tv.kind='text_vector' AND tv.state IN ('ready','processing')
    WHERE f.project_id=? AND f.id>? AND f.id<=?
      AND (av.state IS NOT NULL OR tv.state IS NOT NULL OR f.peaks_json IS NOT NULL)
    ORDER BY f.id LIMIT ?'''


def run(data, *, apply=False, batch_size=128):
    data = Path(data).resolve()
    database = data / 'db' / 'soundmind.db'
    if not database.is_file():
        raise RuntimeError(f'Database does not exist: {database}')
    require_offline()
    os.environ['SOUNDBOT_USER_DATA_DIR'] = str(data)
    # Deliberately avoid DatabaseManager: it recovers jobs/artifacts on startup.
    conn = sqlite3.connect(database.as_uri() + ('?mode=rw' if apply else '?mode=ro'), uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    journal = None
    report = {'apply': apply, 'candidates': 0, 'files_updated': 0, 'vectors_updated': 0, 'failed': 0}
    try:
        projects = [row[0] for row in conn.execute('SELECT id FROM projects')]
        if apply:
            from core.audio_service import get_audio_service
            from core.indexer import get_chroma_client
            service = get_audio_service()
            journal = sqlite3.connect(data / 'metadata-repair-progress.sqlite')
            journal.executescript('''CREATE TABLE IF NOT EXISTS progress
                (project TEXT PRIMARY KEY,after_id INTEGER,high_water INTEGER,complete INTEGER);
                CREATE TABLE IF NOT EXISTS failures
                (project TEXT,file_id INTEGER,error TEXT,PRIMARY KEY(project,file_id));''')
        for project in projects:
            high = conn.execute('SELECT COALESCE(MAX(id),0) FROM files WHERE project_id=?', (project,)).fetchone()[0]
            after = 0
            collections = {}
            retries = []
            if apply:
                progress = journal.execute('SELECT after_id,high_water,complete FROM progress WHERE project=?', (project,)).fetchone()
                if progress and not progress[2]:
                    after, high = progress[:2]
                    retries = [row[0] for row in journal.execute('SELECT file_id FROM failures WHERE project=? AND file_id<=? ORDER BY file_id', (project, after))]
                # A completed run starts a fresh pass, also covering prior failures.
                with journal:
                    journal.execute('INSERT OR REPLACE INTO progress VALUES (?,?,?,0)', (project, after, high))
                if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', project):
                    raise RuntimeError(f'Unsafe project ID: {project!r}')
                chroma_path = (data / 'chroma_projects' / project).resolve()
                chroma_path.relative_to(data / 'chroma_projects')
                client = get_chroma_client(str(chroma_path)) if (chroma_path / 'chroma.sqlite3').is_file() else None
                existing = {c if isinstance(c, str) else c.name for c in client.list_collections()} if client else set()
                for kind, default in [('audio_vector','audio_embeddings'),('text_vector','text_metadata_embeddings')]:
                    manifest = conn.execute('SELECT collection_name FROM index_manifests WHERE project_id=? AND kind=?', (project, kind)).fetchone()
                    name = (manifest[0] if manifest and manifest[0] else default)
                    collections[kind] = client.get_collection(name, embedding_function=None) if name in existing else None
            # Retry each previous error once, then continue the interrupted pass.
            for retry_id in [*retries, None]:
                scan_after, scan_high = (retry_id-1, retry_id) if retry_id is not None else (after, high)
                while scan_after < scan_high:
                    rows = conn.execute(SELECT_ROWS, (project, scan_after, scan_high, batch_size)).fetchall()
                    if not rows:
                        break
                    report['candidates'] += len(rows)
                    if apply:
                        result = repair_page(conn, rows, collections, service)
                        for key in ('files_updated', 'vectors_updated'):
                            report[key] += result[key]
                        with journal:
                            for row in rows:
                                error = result['failures'].get(row['id'])
                                if error:
                                    journal.execute('INSERT OR REPLACE INTO failures VALUES (?,?,?)', (project,row['id'],error))
                                else:
                                    journal.execute('DELETE FROM failures WHERE project=? AND file_id=?', (project,row['id']))
                            if retry_id is None:
                                journal.execute('UPDATE progress SET after_id=? WHERE project=?', (rows[-1]['id'],project))
                        print(json.dumps(dict(report, project=project, after_id=rows[-1]['id'])), flush=True)
                    scan_after = rows[-1]['id']
            if apply:
                with journal:
                    journal.execute('UPDATE progress SET complete=1 WHERE project=?', (project,))
        if apply:
            report['failed'] = journal.execute('SELECT COUNT(*) FROM failures').fetchone()[0]
        print(json.dumps(report), flush=True)
        return report
    finally:
        conn.close()
        if journal:
            journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user-data-dir', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 500:
        parser.error('--batch-size must be between 1 and 500')
    report = run(args.user_data_dir, apply=args.apply, batch_size=args.batch_size)
    if report['failed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
