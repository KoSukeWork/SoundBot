from __future__ import annotations
import asyncio
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
os.environ.setdefault('SOUNDBOT_USER_DATA_DIR', str(ROOT / '.test-data' / 'storage-config'))

from core import data_migration as migration
from core.storage_paths import resolve_data_dir, storage_locks
from core.database import DatabaseManager, AudioFileRecord
import config


def seed(root):
    manager = DatabaseManager(str(root / 'db' / 'soundmind.db'))
    file_id = manager.upsert_file(AudioFileRecord(path=str(root / 'tone.wav'), filename='tone.wav'))
    manager.set_artifact_state('default', file_id, 'text_vector', 'ready', engine_fingerprint='test')
    manager._get_connection().close()
    chroma = root / 'chroma_projects' / 'default'
    chroma.mkdir(parents=True)
    with closing(sqlite3.connect(chroma / 'chroma.sqlite3')) as conn, conn:
        conn.execute('CREATE TABLE test_vectors(id TEXT PRIMARY KEY, embedding BLOB)')
        conn.execute('INSERT INTO test_vectors VALUES (?,?)',(file_id,b'fake-vector'))
    (chroma / 'segment').mkdir()
    (chroma / 'segment' / 'data_level0.bin').write_bytes(bytes(range(256))*40)
    with closing(sqlite3.connect(root / 'metadata-repair-progress.sqlite')) as conn, conn:
        conn.execute('CREATE TABLE progress(id INTEGER)')
        conn.execute('INSERT INTO progress VALUES (12)')
    return file_id


class DataDirectoryTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / '.test-data'
        scratch.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='storage-',dir=scratch))
        self.user, self.target = self.root/'profile', self.root/'new-store'
        self.user.mkdir()
        self.file_id = seed(self.user)
        self.before = {'tempClipDir':str(self.root/'clips'),'unrelated':'保留'}
        (self.user/'user_config.json').write_text(json.dumps(self.before),encoding='utf-8')
        (self.user/'ai_config.json').write_text('{"test":"settings-stay"}')
        self.env = mock.patch.dict(os.environ,{'SOUNDBOT_USER_DATA_DIR':str(self.user)})
        self.env.start()
        self.old_override = os.environ.pop('SOUNDBOT_DATA_DIR',None)
        self.guard = mock.patch.object(migration,'require_backend_offline')
        self.offline = self.guard.start()

    def tearDown(self):
        self.guard.stop()
        self.env.stop()
        # Synthetic artifacts are intentionally retained for inspection.

    def test_default_and_priority_keep_configuration_in_user_data(self):
        self.assertEqual(resolve_data_dir(self.user,{}),self.user)
        self.assertEqual(config.get_db_path(),self.user/'db')
        self.assertEqual(config.get_chroma_db_path(),self.user/'chroma_projects'/'default')
        saved = self.root/'configured'
        (self.user/'user_config.json').write_text(json.dumps(dict(self.before,dataDir=str(saved))))
        self.assertEqual(config.get_data_dir(),saved)
        with mock.patch.dict(os.environ,{'SOUNDBOT_DATA_DIR':str(self.target)}):
            self.assertEqual(config.get_data_dir(),self.target)
            self.assertEqual(config.get_user_data_dir(),self.user)
        (self.user/'user_config.json').write_text('{"dataDir":"relative"}')
        with self.assertRaises(ValueError): resolve_data_dir(self.user,{})

    def test_copy_verify_archive_and_configuration_preservation(self):
        before = migration.inventory(self.user,migration.ITEMS)
        result = migration.migrate(self.user,self.target)
        self.assertEqual(result['phase'],'complete')
        self.assertEqual(migration.inventory(self.target,migration.ITEMS),before)
        self.assertFalse((self.user/'db').exists())
        self.assertTrue(Path(result['archives']['db']).is_dir())
        self.assertTrue((self.user/migration.MOVED).is_file())
        self.assertEqual(json.loads((self.user/'user_config.json').read_text()),dict(self.before,dataDir=str(self.target)))
        self.assertTrue((self.user/'ai_config.json').exists())
        self.assertFalse((self.target/'ai_config.json').exists())
        self.assertTrue((self.target/'metadata-repair-progress.sqlite').exists())
        self.assertEqual(config.get_db_path(),self.target/'db')
        self.assertEqual(config.get_chroma_db_path(),self.target/'chroma_projects'/'default')
        self.assertFalse((self.user/migration.PENDING).exists())

    def test_failure_after_archive_restores_originals_and_cleans_only_own_copy(self):
        self.target.mkdir()
        (self.target/'unrelated.txt').write_text('keep')
        before=migration.inventory(self.user,migration.ITEMS)
        original=migration.write_json
        def fail_config(path,value):
            if Path(path).name=='user_config.json': raise OSError('config write failed')
            return original(path,value)
        with mock.patch.object(migration,'write_json',side_effect=fail_config):
            with self.assertRaisesRegex(OSError,'config write failed'):
                migration.migrate(self.user,self.target)
        self.assertEqual(migration.inventory(self.user,migration.ITEMS),before)
        self.assertEqual(json.loads((self.user/'user_config.json').read_text()),self.before)
        self.assertEqual((self.target/'unrelated.txt').read_text(),'keep')
        self.assertFalse((self.target/'db').exists())
        self.assertEqual(list(self.target.glob('.soundbot-move-*')),[])

    def test_hash_mismatch_rolls_back_before_configuration(self):
        def corrupt(event):
            if event['phase']=='verifying':
                stage=next(self.target.glob('.soundbot-move-*'))
                with (stage/'chroma_projects'/'default'/'segment'/'data_level0.bin').open('ab') as stream:
                    stream.write(b'corrupt')
        with self.assertRaisesRegex(RuntimeError,'SHA-256'):
            migration.migrate(self.user,self.target,progress=corrupt)
        self.assertTrue((self.user/'db'/'soundmind.db').exists())
        self.assertFalse((self.target/'db').exists())
        self.assertEqual(resolve_data_dir(self.user,{}),self.user)

    def test_existing_destination_rejected_without_overwrite(self):
        self.target.mkdir(); seed(self.target)
        before=migration.inventory(self.target,migration.ITEMS)
        with self.assertRaises(FileExistsError): migration.migrate(self.user,self.target)
        self.assertEqual(migration.inventory(self.target,migration.ITEMS),before)
        result=migration.migrate(self.user,self.target,use_existing=True)
        self.assertEqual(result['mode'],'existing')
        self.assertEqual(migration.inventory(self.target,migration.ITEMS),before)

    def test_space_nesting_permissions_and_running_backend_are_rejected(self):
        for target in (self.user,self.user/'nested',self.root):
            with self.assertRaises(ValueError): migration.migrate(self.user,target)
        with mock.patch.object(migration.shutil,'disk_usage',return_value=mock.Mock(free=1)):
            with self.assertRaisesRegex(RuntimeError,'Insufficient'): migration.migrate(self.user,self.target)
        with mock.patch.object(migration.os,'access',return_value=False):
            with self.assertRaises(PermissionError): migration.migrate(self.user,self.target)
        self.offline.side_effect=RuntimeError('running backend')
        with mock.patch.object(migration,'inventory') as reads:
            with self.assertRaisesRegex(RuntimeError,'running'): migration.migrate(self.user,self.target)
            reads.assert_not_called()

    def test_dry_run_does_not_create_target_or_write_configuration(self):
        before=(self.user/'user_config.json').read_bytes()
        result=migration.migrate(self.user,self.target,dry_run=True)
        self.assertEqual(result['phase'],'dry_run')
        self.assertFalse(self.target.exists())
        self.assertEqual((self.user/'user_config.json').read_bytes(),before)

    def test_persistent_change_rejects_environment_override(self):
        with mock.patch.dict(os.environ,{'SOUNDBOT_DATA_DIR':str(self.user)}):
            with self.assertRaisesRegex(ValueError,'Unset SOUNDBOT_DATA_DIR'):
                migration.migrate(self.user,self.target)

    def test_completed_migration_can_roll_back_without_deleting_new_copy(self):
        result=migration.migrate(self.user,self.target)
        migration.recover(self.user,transaction_id=result['id'])
        self.assertEqual(resolve_data_dir(self.user,{}),self.user)
        self.assertTrue((self.user/'db'/'soundmind.db').exists())
        self.assertTrue((self.target/('db.rolled-back-'+result['id'])).is_dir())
        self.assertFalse((self.user/migration.MOVED).exists())

    def test_interrupted_archive_can_be_recovered_from_durable_journal(self):
        original=migration.write_json
        def fail_config(path,value):
            if Path(path).name=='user_config.json': raise OSError('interruption')
            return original(path,value)
        with mock.patch.object(migration,'write_json',side_effect=fail_config), \
             mock.patch.object(migration,'_rollback',side_effect=RuntimeError('process vanished')):
            with self.assertRaisesRegex(RuntimeError,'process vanished'):
                migration.migrate(self.user,self.target)
        self.assertTrue((self.user/migration.PENDING).exists())
        migration.recover(self.user)
        self.assertEqual(resolve_data_dir(self.user,{}),self.user)
        self.assertTrue((self.user/'db'/'soundmind.db').exists())
        self.assertFalse((self.target/'db').exists())

    def test_lock_prevents_second_backend_or_migration(self):
        with storage_locks(self.user):
            with self.assertRaisesRegex(RuntimeError,'in use'):
                migration.migrate(self.user,self.target)

    def test_destination_conflict_after_intent_is_preserved(self):
        original = migration._save
        def create_conflict(user, record):
            original(user, record)
            if record['phase'] == 'installing' and not record.get('rollback_completed') and record['installed'] == ['db']:
                conflict = self.target / 'db'
                conflict.mkdir(exist_ok=True)
                (conflict / 'unrelated.txt').write_text('keep')
        with mock.patch.object(migration, '_save', side_effect=create_conflict):
            with self.assertRaises(OSError):
                migration.migrate(self.user, self.target)
        self.assertEqual((self.target / 'db' / 'unrelated.txt').read_text(), 'keep')
        self.assertTrue((self.user / 'db' / 'soundmind.db').exists())
        self.assertEqual(resolve_data_dir(self.user, {}), self.user)

    def test_interruption_after_staging_removed_recovers(self):
        original = migration.os.replace
        def interrupt_finish(source, target):
            if Path(source).name == migration.PENDING:
                raise OSError('interrupted final journal move')
            return original(source, target)
        with mock.patch.object(migration.os, 'replace', side_effect=interrupt_finish), \
             mock.patch.object(migration, '_rollback', side_effect=RuntimeError('process vanished')):
            with self.assertRaisesRegex(RuntimeError, 'process vanished'):
                migration.migrate(self.user, self.target)
        self.assertEqual(list(self.target.glob('.soundbot-move-*')), [])
        migration.recover(self.user)
        self.assertEqual(resolve_data_dir(self.user, {}), self.user)
        self.assertTrue((self.user / 'db' / 'soundmind.db').exists())
        self.assertFalse((self.target / 'db').exists())

    def test_interrupted_completed_rollback_retains_copy_on_recovery(self):
        result = migration.migrate(self.user, self.target)
        original = migration.atomic_bytes
        def interrupt_config(path, value):
            if Path(path).name == 'user_config.json':
                raise OSError('interrupted rollback')
            return original(path, value)
        with mock.patch.object(migration, 'atomic_bytes', side_effect=interrupt_config):
            with self.assertRaisesRegex(OSError, 'interrupted rollback'):
                migration.recover(self.user, transaction_id=result['id'])
        self.assertTrue((self.user / migration.PENDING).exists())
        migration.recover(self.user)
        self.assertEqual(resolve_data_dir(self.user, {}), self.user)
        self.assertTrue((self.target / ('db.rolled-back-' + result['id'])).is_dir())
        self.assertTrue((self.user / 'db' / 'soundmind.db').exists())

    def test_completed_rollback_cannot_replace_a_pending_transaction(self):
        result = migration.migrate(self.user, self.target)
        pending = self.user / migration.PENDING
        pending.write_text('{"pending":"must survive"}')
        with self.assertRaisesRegex(RuntimeError, '--recover'):
            migration.recover(self.user, transaction_id=result['id'])
        self.assertEqual(pending.read_text(), '{"pending":"must survive"}')
        self.assertEqual(resolve_data_dir(self.user, {}), self.target)


class StorageControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_auth_busy_gate_and_resume(self):
        import httpx
        import main
        root=Path(tempfile.mkdtemp(prefix='storage-control-',dir=ROOT/'.test-data'))
        db=DatabaseManager(str(root/'test.db'))
        server=mock.Mock(should_exit=False)
        try:
            with (mock.patch.dict(os.environ,{'SOUNDBOT_CONTROL_TOKEN':'test-control'}),
                  mock.patch.object(main,'get_db_manager',return_value=db),
                  mock.patch.object(main,'_uvicorn_server',server),
                  mock.patch.object(main,'_storage_shutdown',False),
                  mock.patch.object(main,'_shutting_down',False),
                  mock.patch.object(main,'_startup_paused',False),
                  mock.patch.object(main,'_automatic_index_tasks',{}),
                  mock.patch.object(main,'_worker_futures',set())):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),base_url='http://test') as client:
                    self.assertEqual((await client.post('/api/v1/system/shutdown')).status_code,403)
                    job=db.create_job('default','index_test')
                    headers={'X-SoundBot-Control':'test-control'}
                    self.assertEqual((await client.post('/api/v1/system/shutdown',headers=headers)).status_code,409)
                    self.assertFalse(server.should_exit)
                    db.update_job(job,state='completed')
                    self.assertEqual((await client.post('/api/v1/system/shutdown',headers=headers)).status_code,200)
                    self.assertTrue(server.should_exit)
                    self.assertEqual((await client.post('/api/v1/unused')).status_code,503)
        finally:
            db._get_connection().close()
