from __future__ import annotations

import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('SOUNDBOT_USER_DATA_DIR', str(ROOT / '.test-data' / 'metadata-tests-config'))

from core.audio_service import get_audio_service
from core.database import AudioFileRecord, DatabaseManager
from core.indexer import AudioIndexer, TextMetadataIndexer, build_metadata_text
from core.index_resume import resume_artifacts
import repair_audio_metadata as repair


class Provider:
    fingerprint = 'test-text'
    async def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


class MetadataRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        scratch = ROOT / '.test-data'
        scratch.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='metadata-', dir=scratch))
        self.storage_env = mock.patch.dict(os.environ, {'SOUNDBOT_DATA_DIR': str(self.root)})
        self.storage_env.start()
        self.addCleanup(self.storage_env.stop)
        self.db = DatabaseManager(str(self.root / 'db' / 'soundmind.db'))
        self.service = get_audio_service()
        self.ids = []
        for i in range(3):
            path = self.root / f'stereo-{i}.wav'
            with wave.open(str(path), 'wb') as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(96000)
                wav.writeframes(b'\x10\x00\x20\x00' * 48000)
            self.ids.append(self.db.upsert_file(AudioFileRecord(
                path=str(path), filename=path.name,
                source_fingerprint=self.service.fingerprint(path).source_key,
            )))
        chroma = str(self.root / 'chroma_projects' / 'default')
        self.audio = AudioIndexer(chroma, project_id='default')
        self.text = TextMetadataIndexer(chroma, project_id='default', provider=Provider())
        self.embedder = mock.Mock(fingerprint='test-audio')
        self.embedder.audio_to_embedding.return_value = [1., 0.]
        self.probes = Counter()
        self.stop = False
        self.fake_service = mock.Mock(waveform=self.service.waveform)
        def probe(path):
            self.probes[path] += 1
            return self.service.probe(path)
        self.fake_service.probe = probe

    def tearDown(self):
        self.db._get_connection().close()
        # Retain only synthetic data for inspection; Chroma owns Windows handles.

    async def resume(self, **kwargs):
        async def inline(fn, *args, **kw):
            return fn(*args, **kw)
        job = self.db.create_job('default', 'index_test')
        with mock.patch('core.indexer.get_embedder', return_value=self.embedder):
            result = await resume_artifacts(
                self.db, 'default', job, ['audio_vector','text_vector','waveform'],
                {'audio_vector': self.audio, 'text_vector': self.text}, self.fake_service,
                run_audio=inline, run_model=inline,
                build_text=lambda row: dict(row, file_id=row['file_uuid'],file_path=row['path']),
                model_available=lambda: True, shutting_down=lambda: self.stop,
                batch_size=2, **kwargs)
        return job, result

    async def test_metadata_precedes_audio_and_text_and_only_one_probe_per_file(self):
        _, (failures, cancelled) = await self.resume()
        self.assertFalse(cancelled)
        self.assertEqual(sum(failures.values()), 0)
        self.assertEqual(list(self.probes.values()), [1,1,1])
        for file_id in self.ids:
            record = self.db.get_file_by_uuid(file_id, 'default')
            self.assertEqual((record.duration,record.sample_rate,record.channels),(.5,96000,2))
            for indexer in (self.audio,self.text):
                meta = indexer.collection.get(ids=[file_id],include=['metadatas'])['metadatas'][0]
                self.assertEqual((meta['duration'],meta['sample_rate'],meta['channels']),(.5,96000,2))
            for kind in ('waveform','audio_vector','text_vector'):
                self.assertEqual(self.db.get_artifact('default',file_id,kind)['state'],'ready')
        calls = self.embedder.audio_to_embedding.call_count
        await self.resume()
        self.assertEqual(self.embedder.audio_to_embedding.call_count,calls)

    async def test_batch_vector_failure_then_restart_does_not_redo_ready_artifacts(self):
        original = self.audio.write_prepared_audio
        with mock.patch.object(self.audio,'write_prepared_audio',side_effect=RuntimeError('disk failure')):
            _, (failures, _) = await self.resume()
        self.assertEqual(failures, {'audio_vector':3,'text_vector':0,'waveform':0})
        self.assertEqual(self.embedder.audio_to_embedding.call_count,3)
        before = self.db.get_artifact('default',self.ids[0],'waveform')
        with mock.patch.object(self.audio,'write_prepared_audio',wraps=original), \
             mock.patch.object(self.text,'prepare_metadata',side_effect=AssertionError('ready text rewritten')):
            _, (failures, _) = await self.resume()
        self.assertEqual(sum(failures.values()),0)
        self.assertEqual(before,self.db.get_artifact('default',self.ids[0],'waveform'))

    async def test_one_bad_text_does_not_poison_other_files_in_batch(self):
        original = self.text.provider.embed_texts
        attempted = []
        async def encode(texts):
            attempted.extend(texts)
            if 'stereo-1.wav' in texts[0]:
                raise RuntimeError('bad text')
            return await original(texts)
        with mock.patch.object(self.text.provider,'embed_texts',side_effect=encode):
            _, (failures, _) = await self.resume()
        self.assertEqual(failures,{'audio_vector':0,'text_vector':1,'waveform':0})
        self.assertEqual(len(attempted),3)
        self.assertEqual(self.text.collection.count(),2)
        with mock.patch.object(self.text.provider,'embed_texts',wraps=original) as retry:
            await self.resume()
        self.assertEqual(retry.await_count,1)

    async def test_pause_flushes_prepared_file_and_restores_unstarted_claims(self):
        self.db.set_artifact_state('default',self.ids[1],'text_vector','failed',error_code='old_failure')
        original = self.fake_service.waveform
        def waveform(path):
            self.stop = True
            return original(path)
        with mock.patch.object(self.fake_service,'waveform',side_effect=waveform):
            job, (_, cancelled) = await self.resume()
        self.assertTrue(cancelled)
        self.assertEqual(self.db.get_job(job)['processed'],3)
        self.assertEqual(self.db.get_artifact('default',self.ids[1],'text_vector')['error_code'],'old_failure')
        self.assertTrue(all(counts['processing']==0 for counts in self.db.get_artifact_counts('default').values()))
        self.stop = False
        await self.resume()
        self.assertEqual(self.embedder.audio_to_embedding.call_count,3)
        self.assertEqual(self.text.collection.count(),3)

    async def test_one_bad_chroma_row_does_not_fail_other_rows(self):
        original = self.audio.write_prepared_audio
        bad = self.ids[1]
        def write(rows):
            if any(row['file_id'] == bad for row in rows):
                raise ValueError('one invalid Chroma row')
            return original(rows)
        with mock.patch.object(self.audio,'write_prepared_audio',side_effect=write):
            _,(failures,cancelled)=await self.resume()
        self.assertFalse(cancelled)
        self.assertEqual(failures,{'audio_vector':1,'text_vector':0,'waveform':0})
        self.assertEqual(self.audio.collection.count(),2)
        self.assertEqual(self.db.get_artifact('default',bad,'audio_vector')['state'],'failed')

    async def test_failed_probe_is_attempted_once_per_file_and_pass(self):
        original = self.fake_service.probe
        attempts = []
        def probe(path):
            attempts.append(path)
            if path.endswith('stereo-1.wav'):
                raise RuntimeError('missing audio')
            return original(path)
        with mock.patch.object(self.fake_service,'probe',side_effect=probe):
            _, (failures, _) = await self.resume()
        self.assertEqual(failures,{'audio_vector':1,'text_vector':1,'waveform':1})
        self.assertEqual(len(attempts),3)
        self.assertEqual(self.audio.collection.count(),2)

    async def test_fatal_publication_restores_claims_without_restart(self):
        with mock.patch('core.index_resume.publish_batch',side_effect=RuntimeError('publish failure')):
            with self.assertRaisesRegex(RuntimeError,'publish failure'):
                await self.resume()
        self.assertEqual(self.db.get_file_by_uuid(self.ids[0],'default').duration,0)
        self.assertEqual(self.audio.collection.count(),2)
        self.assertTrue(all(counts['processing']==0 for counts in self.db.get_artifact_counts('default').values()))
        await self.resume()
        self.assertEqual(self.audio.collection.count(),3)
        self.assertEqual(self.text.collection.count(),3)
        self.assertEqual(self.db.get_artifact_counts('default')['audio_vector']['ready'],3)

    async def test_waveform_no_matching_path_only_fails_one_artifact(self):
        with self.db.get_cursor() as cur:
            cur.execute("UPDATE files SET canonical_path='mismatched/path' WHERE file_uuid=?",(self.ids[0],))
        job, (failures, _) = await self.resume()
        self.assertEqual(failures,{'waveform':1,'audio_vector':0,'text_vector':0})
        self.assertEqual(self.db.get_job(job)['processed'],9)
        failed=self.db.get_artifact('default',self.ids[0],'waveform')
        self.assertIn('matched no file',failed['error_message'])
        self.assertEqual(self.db.get_artifact_counts('default')['waveform']['ready'],2)
        self.assertEqual(self.db.get_artifact_counts('default')['audio_vector']['ready'],3)

    async def test_invalid_peaks_rolls_back_only_its_waveform(self):
        from dataclasses import replace
        original=self.fake_service.waveform
        def waveform(path):
            result=original(path)
            return replace(result,peaks=[float('nan')]) if path.endswith('stereo-0.wav') else result
        with mock.patch.object(self.fake_service,'waveform',side_effect=waveform):
            _,(failures,_)=await self.resume()
        self.assertEqual(failures,{'waveform':1,'audio_vector':0,'text_vector':0})
        self.assertIn('finite',self.db.get_artifact('default',self.ids[0],'waveform')['error_message'])
        self.assertEqual(self.db.get_artifact_counts('default')['waveform']['ready'],2)

    async def test_transient_sqlite_publication_retries_batch_once(self):
        original=self.db.update_audio_metadata
        attempts=Counter()
        def update(project,file_id,metadata):
            attempts[file_id]+=1
            if file_id==self.ids[0] and attempts[file_id]==1:
                raise sqlite3.OperationalError('transient write failure')
            return original(project,file_id,metadata)
        with mock.patch.object(self.db,'update_audio_metadata',side_effect=update):
            _,(failures,_)=await self.resume()
        self.assertEqual(sum(failures.values()),0)
        self.assertEqual(attempts[self.ids[0]],2)
        self.assertEqual(self.embedder.audio_to_embedding.call_count,3)

    async def test_persistent_sqlite_row_error_isolated_after_two_batch_attempts(self):
        # Real SQLite trigger exercises rollback, retry and the per-row path.
        with self.db.get_cursor() as cur:
            cur.execute(f"""CREATE TRIGGER reject_one_text BEFORE UPDATE ON file_artifacts
                WHEN NEW.file_uuid='{self.ids[1]}' AND NEW.kind='text_vector' AND NEW.state='ready'
                BEGIN SELECT RAISE(ABORT,'one row cannot publish'); END""")
        job,(failures,_)=await self.resume()
        self.assertEqual(failures,{'waveform':0,'audio_vector':0,'text_vector':1})
        self.assertEqual(self.db.get_job(job)['processed'],9)
        self.assertEqual(self.db.get_artifact_counts('default')['text_vector']['ready'],2)
        self.assertIn('one row cannot publish',self.db.get_artifact('default',self.ids[1],'text_vector')['error_message'])
        self.assertTrue(all(c['processing']==0 for c in self.db.get_artifact_counts('default').values()))
        self.assertEqual(self.embedder.audio_to_embedding.call_count,3)

    async def test_task_cancellation_restores_original_failed_claim(self):
        self.db.set_artifact_state('default',self.ids[0],'text_vector','failed',error_code='original_error')
        with mock.patch.object(self.fake_service,'waveform',side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await self.resume()
        self.assertTrue(all(c['processing']==0 for c in self.db.get_artifact_counts('default').values()))
        original=self.db.get_artifact('default',self.ids[0],'text_vector')
        self.assertEqual((original['state'],original['error_code']),('failed','original_error'))

    async def test_commit_release_error_rolls_back_before_retry(self):
        connection=self.db._get_connection()
        class ConnectionProxy:
            releases=0
            def __getattr__(self,name):
                return getattr(connection,name)
            def execute(self,sql,*args):
                if sql=='RELEASE soundbot_batch_0':
                    self.releases+=1
                    if self.releases==2:
                        raise sqlite3.OperationalError('commit I/O interrupted')
                return connection.execute(sql,*args)
        proxy=ConnectionProxy()
        with mock.patch.object(self.db,'_get_connection',return_value=proxy):
            _,(failures,_)=await self.resume()
        self.assertEqual(sum(failures.values()),0)
        self.assertFalse(connection.in_transaction)
        self.assertEqual(self.db.get_artifact_counts('default')['waveform']['ready'],3)

    async def test_fatal_mid_fallback_preserves_published_rows_and_restores_remaining(self):
        original=self.db.set_artifact_state
        attempts=0
        def publish(project,file_id,kind,state,**kwargs):
            nonlocal attempts
            if file_id==self.ids[1] and kind=='text_vector' and state=='ready':
                attempts+=1
                if attempts<=2:
                    raise sqlite3.OperationalError('batch fails')
                raise RuntimeError('unexpected fatal fallback')
            return original(project,file_id,kind,state,**kwargs)
        with mock.patch.object(self.db,'set_artifact_state',side_effect=publish):
            with self.assertRaisesRegex(RuntimeError,'fatal fallback'):
                await self.resume()
        self.assertEqual(self.db.get_artifact('default',self.ids[0],'audio_vector')['state'],'ready')
        self.assertEqual(self.db.get_artifact('default',self.ids[1],'text_vector')['state'],'pending')
        self.assertTrue(all(c['processing']==0 for c in self.db.get_artifact_counts('default').values()))
        await self.resume()
        self.assertEqual(self.embedder.audio_to_embedding.call_count,3)

    async def test_job_finishes_entire_pass_despite_one_waveform_save_failure(self):
        import main
        with self.db.get_cursor() as cur:
            cur.execute("UPDATE files SET canonical_path='missing/path' WHERE file_uuid=?",(self.ids[0],))
        async def inline(fn,*args,**kwargs):
            return fn(*args,**kwargs)
        job=self.db.create_job('default','index_auto_publish_test')
        with (mock.patch.object(main,'get_db_manager',return_value=self.db),
              mock.patch.object(main,'_get_active_audio_indexer',return_value=self.audio),
              mock.patch.object(main,'_get_active_text_indexer',return_value=self.text),
              mock.patch.object(main,'_run_audio_work',side_effect=inline),
              mock.patch.object(main,'_run_model_work',side_effect=inline),
              mock.patch.object(main,'is_embedder_available',return_value=True),
              mock.patch.object(main,'_shutting_down',False),
              mock.patch('core.indexer.get_embedder',return_value=self.embedder),
              mock.patch('core.audio_service.get_audio_service',return_value=self.fake_service)):
            await main._repair_index_task_locked(job,'default',['waveform','audio_vector','text_vector'],'resume')
        status=self.db.get_job(job)
        self.assertEqual((status['stage'],status['processed'],status['total']),('done_with_errors',9,9))
        self.assertTrue(all(c['processing']==0 for c in self.db.get_artifact_counts('default').values()))

    async def test_offline_repair_preserves_vectors_metadata_and_readiness_and_is_repeatable(self):
        await self.resume()
        # One file needs both stores fixed; another has correct SQLite but bad Chroma.
        with self.db.get_cursor() as cur:
            cur.execute('UPDATE files SET duration=0,sample_rate=0,channels=0 WHERE file_uuid=?',(self.ids[0],))
        originals = {}
        for indexer in (self.audio,self.text):
            indexer.collection.update(ids=self.ids[:2],metadatas=[{'duration':0,'sample_rate':0,'channels':0,'sentinel':'keep'}]*2)
            originals[indexer.collection_name] = indexer.collection.get(include=['embeddings','metadatas'])
        states = self.db.get_artifact_counts('default')
        with mock.patch.object(repair,'require_offline'):
            first = repair.run(self.root,apply=True,batch_size=1)
            second = repair.run(self.root,apply=True,batch_size=1)
        self.assertEqual((first['files_updated'],first['vectors_updated'],first['failed']),(1,4,0))
        self.assertEqual((second['files_updated'],second['vectors_updated'],second['failed']),(0,0,0))
        self.assertEqual(states,self.db.get_artifact_counts('default'))
        for indexer in (self.audio,self.text):
            after = indexer.collection.get(include=['embeddings','metadatas'])
            before = originals[indexer.collection_name]
            self.assertEqual(after['ids'],before['ids'])
            self.assertTrue((after['embeddings']==before['embeddings']).all())
            for old,new in zip(before['metadatas'],after['metadatas']):
                for key in set(old)-{'duration','sample_rate','channels'}:
                    self.assertEqual(old[key],new[key])
        self.assertEqual(build_metadata_text({'filename':'a.wav','duration':0}),
                         build_metadata_text({'filename':'a.wav','duration':.5,'sample_rate':96000,'channels':2}))

    async def test_offline_crash_before_checkpoint_retries_completed_page(self):
        await self.resume()
        for indexer in (self.audio,self.text):
            indexer.collection.update(ids=self.ids,metadatas=[{'duration':0}]*3)
        real = repair.repair_page
        calls = 0
        def crash_after_write(*args):
            nonlocal calls
            result = real(*args)
            calls += 1
            if calls == 2:
                raise RuntimeError('power loss before checkpoint')
            return result
        with mock.patch.object(repair,'require_offline'), mock.patch.object(repair,'repair_page',side_effect=crash_after_write):
            with self.assertRaisesRegex(RuntimeError,'power loss'):
                repair.run(self.root,apply=True,batch_size=1)
        with mock.patch.object(repair,'require_offline'):
            result = repair.run(self.root,apply=True,batch_size=1)
        self.assertEqual(result['candidates'],2)
        self.assertEqual(result['vectors_updated'],2)
        self.assertEqual(result['failed'],0)

    async def test_offline_missing_audio_is_journaled_and_retried(self):
        await self.resume()
        with self.db.get_cursor() as cur:
            cur.execute('UPDATE files SET duration=0 WHERE file_uuid=?',(self.ids[0],))
        with mock.patch.object(repair,'require_offline'), \
             mock.patch('core.audio_service.get_audio_service',return_value=mock.Mock(probe=mock.Mock(side_effect=RuntimeError('missing file')))):
            result = repair.run(self.root,apply=True)
        self.assertEqual(result['failed'],1)
        with mock.patch.object(repair,'require_offline'):
            result = repair.run(self.root,apply=True)
        self.assertEqual(result['failed'],0)

    def test_offline_guard_prevents_any_database_open(self):
        with mock.patch.object(repair,'require_offline',side_effect=RuntimeError('running')), \
             mock.patch.object(repair.sqlite3,'connect') as connect:
            with self.assertRaisesRegex(RuntimeError,'running'):
                repair.run(self.root,apply=True)
            connect.assert_not_called()


if __name__=='__main__':
    unittest.main()
