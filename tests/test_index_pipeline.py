from __future__ import annotations
import asyncio
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
os.environ.setdefault('SOUNDBOT_USER_DATA_DIR', str(ROOT / '.test-data' / 'pipeline-config'))
from core.audio_service import AudioService, WaveformResult, FileFingerprint
from core.index_audio import prepare_index_audio, file_peaks, IndexWorkCancelled, DecodePool
from core.database import DatabaseManager, AudioFileRecord
from core.index_resume import resume_artifacts


def fixture(root, name='tone.wav', rate=44100, channels=2, frames=44101):
    path = root / name
    values = np.random.default_rng(42).integers(-24000, 24000, (frames, channels), dtype=np.int16)
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(channels); out.setframerate(rate); out.setsampwidth(2)
        out.writeframes(values.astype('<i2').tobytes())
    return path


class PreparedAudioTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='pipeline-', dir=ROOT / '.test-data'))
        self.service = AudioService(self.root / 'cache')

    def test_half_read_window_failure_closes_spool_without_masking_exception(self):
        import torch
        from core.index_audio import IndexAudio
        from core.embedder import CLIPEmbedder
        path = self.root / 'index-audio-abcdefgh.f32'
        np.ones(64, dtype=np.float32).tofile(path)
        prepared = IndexAudio(path, 48000)
        embedder = object.__new__(CLIPEmbedder)
        embedder.max_samples = 4
        embedder._processor_inputs = lambda *a, **kw: {'x': torch.ones(1, 3)}
        # Keep the traceback alive across cleanup, as production logging does.
        for error in (RuntimeError('model failed'), torch.OutOfMemoryError('batch=1'),
                      IndexWorkCancelled()):
            np.ones(64, dtype=np.float32).tofile(path)
            embedder._forward_audio_batch = mock.Mock(side_effect=error)
            caught = None
            with mock.patch.dict(os.environ, {'SOUNDBOT_CLAP_BATCH_SIZE': '1'}):
                try:
                    embedder.prepared_audio_to_embedding(prepared)
                except BaseException as exc:
                    caught = exc
                finally:
                    prepared.close()
            self.assertIs(caught, error)
            self.assertFalse(path.exists())

    def test_locked_spool_cleanup_does_not_mask_original_error(self):
        from core.index_audio import IndexAudio
        prepared = IndexAudio(self.root / 'index-audio-abcdefgh.f32', 48000)
        error = ValueError('original')
        with mock.patch.object(Path, 'unlink', side_effect=PermissionError(32, 'locked')):
            with self.assertRaisesRegex(ValueError, 'original'):
                try:
                    raise error
                finally:
                    prepared.close()

    def test_stale_cleanup_matches_only_owned_regular_old_files(self):
        from core.index_audio import cleanup_stale_spools
        now = time.time()
        names = ('index-audio-abcdefgh.f32', 'index-audio-12345678.f32',
                 'index-audio-user-recording.f32', 'user.wav')
        for name in names:
            path = self.root / name
            path.write_bytes(b'data')
            os.utime(path, (now - 8 * 86400, now - 8 * 86400))
        os.utime(self.root / names[1], (now, now))
        folder = self.root / 'index-audio-abcd1234.f32'
        folder.mkdir()
        os.utime(folder, (now - 8 * 86400, now - 8 * 86400))
        cleanup_stale_spools(self.root, now=now)
        self.assertFalse((self.root / names[0]).exists())
        self.assertTrue(all((self.root / name).exists() for name in names[1:]))
        self.assertTrue(folder.is_dir())

    def test_default_parallelism_tracks_available_cpus_with_a_cap(self):
        from core.index_audio import default_parallelism
        for cpus, expected in ((None, 1), (1, 1), (2, 2), (32, 4)):
            with mock.patch('core.index_audio.os.cpu_count', return_value=cpus):
                self.assertEqual(default_parallelism(), expected)

    def test_single_decode_preserves_exact_waveform_and_resampled_samples(self):
        from core import index_audio
        for rate, channels, frames in ((8000,1,7),(44100,2,50001),(96000,2,192001)):
            path = fixture(self.root, f'{rate}.wav',rate,channels,frames)
            old = self.service._waveform_with_pyav_streaming(path)
            decoded = self.service.decode(path, target_sample_rate=48000, mono=True)
            with mock.patch.object(index_audio._av, 'open', wraps=index_audio._av.open) as opens:
                prepared = prepare_index_audio(self.service, path, 48000)
            self.assertEqual(opens.call_count, 1)
            try:
                self.assertEqual(prepared.waveform, old)
                blocks = list(prepared.windows(10000))
                self.assertTrue(all(len(block)<=10000 for block in blocks))
                np.testing.assert_array_equal(np.concatenate(blocks) if blocks else np.array([],dtype=np.float32),
                                              decoded.samples.reshape(-1))
            finally:
                prepared.close()
            self.assertFalse(prepared.path.exists())

    def test_file_peaks_exact_and_reads_never_exceed_one_mib(self):
        values = np.random.default_rng(7).normal(size=2_000_003).astype(np.float32)
        class Bounded(io.BytesIO):
            def read(self, count=-1):
                self_case.assertGreater(count, 0)
                self_case.assertLessEqual(count, 1024**2)
                return super().read(count)
        self_case = self
        self.assertEqual(file_peaks(Bounded(values.tobytes()),len(values)),
                         self.service._fixed_peaks_bounded(values,2000))

    def test_cancel_during_decode_cleans_owned_spool(self):
        path=fixture(self.root); stop=threading.Event(); stop.set()
        with self.assertRaises(IndexWorkCancelled):
            prepare_index_audio(self.service,path,48000,cancel=stop)
        self.assertEqual(list(self.service.waveform_work_dir.glob('index-audio-*')), [])

    def test_cpu_vectorized_mel_matches_reference_for_short_full_and_tail(self):
        from transformers import ClapFeatureExtractor
        from core.clap_features import window_inputs
        extractor=ClapFeatureExtractor()
        for size in (1, 12345, 480000):
            audio=np.random.default_rng(size).normal(size=size).astype(np.float32)*.1
            reference=extractor([audio],sampling_rate=48000,max_length=480000,
                                truncation='rand_trunc',padding='repeatpad',return_tensors='pt')
            actual=window_inputs(audio,extractor,480000)
            np.testing.assert_array_equal(actual['is_longer'],reference['is_longer'])
            np.testing.assert_allclose(actual['input_features'],reference['input_features'],rtol=0,atol=1e-5)

    def test_real_decode_process_returns_identical_results_and_exits(self):
        path=fixture(self.root)
        reference=prepare_index_audio(self.service,path,48000)
        pool=DecodePool(1)
        try:
            actual=pool.prepare(self.service,str(path),48000,waveform=True)
            self.assertEqual(actual.waveform,reference.waveform)
            self.assertEqual(actual.path.read_bytes(),reference.path.read_bytes())
            actual.close()
            from core.audio_service import AudioServiceError
            with self.assertRaises(AudioServiceError):
                pool.prepare(self.service,str(self.root/'missing.wav'),48000,waveform=True)
            # A failed file must not poison the process result channel.
            actual=pool.prepare(self.service,str(path),48000,waveform=True)
            self.assertEqual(actual.waveform,reference.waveform)
            actual.close()
            pool.cancel.set()
            with self.assertRaises(IndexWorkCancelled):
                pool.prepare(self.service,str(path),48000,waveform=True)
            processes=list(pool.pool._processes.values())
        finally:
            reference.close();pool.close()
        self.assertTrue(all(not process.is_alive() for process in processes))

    def test_oom_splits_batch_without_reordering_or_changing_values(self):
        import torch
        from core.embedder import CLIPEmbedder
        embedder=object.__new__(CLIPEmbedder)
        embedder.device='cpu';embedder._inference_lock=threading.Lock()
        calls=[]
        def forward(input_features):
            calls.append(len(input_features))
            if len(input_features)>2: raise torch.OutOfMemoryError('test pressure')
            return input_features
        embedder.model=mock.Mock(get_audio_features=forward)
        values=torch.arange(20,dtype=torch.float32).reshape(5,4)
        actual=embedder._forward_audio_batch({'input_features':values})
        np.testing.assert_array_equal(actual,values.numpy())
        self.assertEqual(calls,[5,2,3,1,2])

    def test_cancelled_batch_never_enters_model(self):
        import torch
        from core.embedder import CLIPEmbedder
        embedder=object.__new__(CLIPEmbedder)
        embedder.device='cpu';embedder._inference_lock=threading.Lock();embedder.model=mock.Mock()
        stop=threading.Event();stop.set()
        with self.assertRaises(IndexWorkCancelled):
            embedder._forward_audio_batch({'input_features':torch.ones(2,3)},stop)
        embedder.model.get_audio_features.assert_not_called()


class ConcurrentRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='pipeline-resume-',dir=ROOT/'.test-data'))
        self.db=DatabaseManager(str(self.root/'db.sqlite'))
        self.ids=[]
        for i in range(6):
            path=fixture(self.root,f'{i}.wav',frames=32)
            self.ids.append(self.db.upsert_file(AudioFileRecord(path=str(path),filename=path.name,
                duration=1,sample_rate=44100,channels=2)))
        self.job=self.db.create_job('default','test_pipeline')
        self.active=0; self.maximum=0; self.completed=[]; self.lock=threading.Lock()

    def tearDown(self):
        self.db._get_connection().close()

    async def run_job(self, waveform, shutting_down=lambda:False):
        service=AudioService(self.root/'cache')
        service.waveform=waveform
        with mock.patch.dict(os.environ,{'SOUNDBOT_INDEX_PREFETCH':'2'}):
            return await resume_artifacts(self.db,'default',self.job,['waveform'],{},service,
                run_audio=asyncio.to_thread,run_model=asyncio.to_thread,build_text=lambda r:r,
                model_available=lambda:False,shutting_down=shutting_down)

    def work(self,path,*,cancel=None):
        number=int(Path(path).stem)
        with self.lock:
            self.active+=1;self.maximum=max(self.maximum,self.active)
        try:
            for _ in range(10):
                if cancel.is_set(): raise IndexWorkCancelled()
                time.sleep(.005 if number%2 else .01)
            if number==2: raise ValueError('bad file only')
            self.completed.append(number)
            return WaveformResult([number/10]*2000,1,44100,FileFingerprint(10,number))
        finally:
            with self.lock:self.active-=1

    async def test_concurrent_results_stay_with_rows_failure_and_ready_are_isolated(self):
        self.db.set_artifact_state('default',self.ids[5],'waveform','ready')
        before=self.db.get_artifact('default',self.ids[5],'waveform')
        failures,cancelled=await self.run_job(self.work)
        self.assertEqual((failures,cancelled),({'waveform':1},False))
        self.assertEqual(self.maximum,2)
        self.assertEqual(self.active,0)
        self.assertEqual(before,self.db.get_artifact('default',self.ids[5],'waveform'))
        for i in (0,1,3,4):
            with self.db.get_cursor() as cursor:
                peaks=cursor.execute('SELECT peaks_json FROM files WHERE file_uuid=?',(self.ids[i],)).fetchone()[0]
            self.assertEqual(json.loads(peaks),[i/10]*2000)
        self.assertEqual(self.db.get_artifact('default',self.ids[2],'waveform')['state'],'failed')

    async def test_cancel_request_stops_workers_and_restores_claims_within_half_second(self):
        entered=threading.Event()
        def blocked(path,*,cancel):
            entered.set()
            while not cancel.wait(.005):pass
            raise IndexWorkCancelled()
        task=asyncio.create_task(self.run_job(blocked))
        await asyncio.to_thread(entered.wait,2)
        start=time.monotonic();self.db.update_job(self.job,cancel_requested=1)
        _,cancelled=await task
        self.assertTrue(cancelled)
        self.assertLess(time.monotonic()-start,.5)
        self.assertEqual(self.db.get_artifact_counts('default')['waveform']['processing'],0)

    async def test_task_cancel_drains_worker_before_restoring_claim_and_restart(self):
        entered=threading.Event();exited=threading.Event()
        def blocked(path,*,cancel):
            entered.set()
            while not cancel.wait(.005):pass
            exited.set();raise IndexWorkCancelled()
        task=asyncio.create_task(self.run_job(blocked))
        await asyncio.to_thread(entered.wait,2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(exited.is_set())
        self.assertEqual(self.db.get_artifact_counts('default')['waveform']['processing'],0)
        self.db.set_artifact_state('default',self.ids[0],'waveform','processing')
        self.db._get_connection().close()
        self.db=DatabaseManager(str(self.root/'db.sqlite'))
        self.assertEqual(self.db.get_artifact('default',self.ids[0],'waveform')['state'],'pending')
        await self.run_job(self.work)
        self.assertEqual(self.db.get_artifact_counts('default')['waveform']['ready'],5)

    async def test_graceful_shutdown_signal_stops_preparation(self):
        entered=threading.Event();shutdown=threading.Event()
        def blocked(path,*,cancel):
            entered.set()
            while not cancel.wait(.005):pass
            raise IndexWorkCancelled()
        task=asyncio.create_task(self.run_job(blocked,shutdown.is_set))
        await asyncio.to_thread(entered.wait,2)
        shutdown.set()
        _,cancelled=await task
        self.assertTrue(cancelled)
        self.assertEqual(self.db.get_artifact_counts('default')['waveform']['processing'],0)

    async def test_task_cancel_during_chroma_waits_before_releasing_claims(self):
        entered=threading.Event();release=threading.Event();finished=threading.Event()
        async def prepare(records):
            return [dict(file_id=r['file_uuid'],embedding=[1.,0.],
                         metadata={'engine_fingerprint':'test'}) for r in records]
        def write(rows):
            entered.set()
            if not release.wait(2):raise TimeoutError('test did not release writer')
            finished.set()
        target=mock.Mock(prepare_metadata=prepare,write_prepared_metadata=write)
        task=asyncio.create_task(resume_artifacts(self.db,'default',self.job,['text_vector'],
            {'text_vector':target},AudioService(self.root/'cache'),run_audio=asyncio.to_thread,
            run_model=asyncio.to_thread,build_text=lambda r:r,model_available=lambda:False,
            shutting_down=lambda:False))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait,2))
            task.cancel();await asyncio.sleep(.03)
            self.assertFalse(task.done())
            self.assertFalse(finished.is_set())
            self.assertEqual(self.db.get_artifact_counts('default')['text_vector']['processing'],6)
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(finished.is_set())
        counts=self.db.get_artifact_counts('default')['text_vector']
        self.assertEqual((counts['processing'],counts['pending']),(0,6))
