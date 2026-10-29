from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import io
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock
import wave

import numpy as np
import torch
from transformers import ClapFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from core.audio_service import AudioService, AudioMetadata, WaveformResult, FileFingerprint
from core.audio_metadata import probe_metadata
from core.index_audio import prepare_index_audio, IndexWorkCancelled
from core.index_features import FeatureSpec, window_features
from core.index_workers import FeaturePool
from core.index_pipeline import resume_continuous
from core.database import DatabaseManager, AudioFileRecord


def make_wave(root, name='tone.wav', rate=48000, channels=1, seconds=.2):
    path = root / name
    data = np.random.default_rng(42).integers(-1000, 1000, (int(rate*seconds), channels), dtype=np.int16)
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(channels);out.setframerate(rate);out.setsampwidth(2);out.writeframes(data.tobytes())
    return path


def spec_embedder():
    return types.SimpleNamespace(max_samples=480000, sample_rate=48000,
                                 processor=types.SimpleNamespace(feature_extractor=ClapFeatureExtractor()))


class FeatureTests(unittest.TestCase):
    def setUp(self):
        (ROOT / '.test-data').mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='r5-feature-',dir=ROOT/'.test-data'))

    def test_numpy_worker_features_match_reference_and_r4_bitwise(self):
        from core.clap_features import window_inputs
        e = spec_embedder();spec=FeatureSpec.from_embedder(e)
        for length in (1, 48123, 480000):
            audio=np.random.default_rng(length).normal(size=length).astype(np.float32)*.2
            actual=window_features(audio,spec)
            old=window_inputs(audio,e.processor.feature_extractor,480000)['input_features'].numpy()
            np.testing.assert_array_equal(actual,old)

    def test_shared_container_metadata_matches_probe_field_by_field_with_one_open(self):
        from core import index_audio
        service=AudioService(self.root/'cache')
        for rate,channels in ((8000,1),(44100,2),(96000,2)):
            path=make_wave(self.root,f'{rate}.wav',rate,channels)
            expected=probe_metadata(service,path)
            with mock.patch.object(index_audio._av,'open',wraps=index_audio._av.open) as opened:
                prepared=prepare_index_audio(service,path,48000,need_metadata=True)
            try:
                self.assertEqual(prepared.metadata.to_dict(),expected.to_dict())
                self.assertEqual(opened.call_count,1)
            finally:prepared.close()

    def test_spool_limit_preserves_error_and_cleans_owned_file(self):
        service=AudioService(self.root/'cache');path=make_wave(self.root)
        with self.assertRaisesRegex(Exception,'spool budget'):
            prepare_index_audio(service,path,48000,max_spool_bytes=16)
        self.assertEqual(list(service.waveform_work_dir.glob('index-audio-*.f32')),[])

    def test_peak_reduction_matches_old_bins_at_read_and_empty_bin_boundaries(self):
        from core.index_audio import file_peaks
        for size in (0, 1, 2, 1999, 2000, 2001, 262143, 262144, 262145, 524289):
            samples = np.random.default_rng(size).normal(size=size).astype(np.float32)
            self.assertEqual(file_peaks(io.BytesIO(samples.tobytes()), size),
                             AudioService._fixed_peaks_bounded(samples, 2000))

    def test_missing_duration_uses_same_metadata_fallback(self):
        service=AudioService(self.root/'cache');path=make_wave(self.root,channels=2)
        header=AudioMetadata(0,48000,2,'wav','pcm_s16le')
        with mock.patch.object(AudioService,'_metadata_from_container',return_value=header):
            expected=probe_metadata(service,path)
            prepared=prepare_index_audio(service,path,48000,need_metadata=True)
        try:self.assertEqual(prepared.metadata,expected)
        finally:prepared.close()

    def test_two_mel_threads_preserve_order_and_bound_outstanding_windows(self):
        from core import index_workers
        entered=[];released=[];peak=[0];active=[0];lock=threading.Lock()
        def timed(value):
            with lock:
                active[0]+=1;peak[0]=max(peak[0],active[0]);entered.append(value)
            time.sleep(.01 if value%2 else .02)
            with lock:active[0]-=1
            return value,0.
        def windows():
            for value in range(8):
                # Two futures plus one value held by the consumer is the bound.
                self.assertLessEqual(value-len(released),2)
                yield value
        with mock.patch.object(index_workers,'_cancel',threading.Event()), mock.patch.object(index_workers,'_timed_features',side_effect=timed), mock.patch.dict(os.environ,{'SOUNDBOT_FEATURE_THREADS':'2'}):
            for value,_ in index_workers._mel_windows(windows()):released.append(value)
        self.assertEqual(released,list(range(8)))
        self.assertEqual(peak[0],2)

    def test_full_feature_queue_can_cancel_without_feeder_deadlock(self):
        spec=FeatureSpec.from_embedder(spec_embedder());spec.max_samples=48000
        service=AudioService(self.root/'cache');path=make_wave(self.root,seconds=60)
        pool=FeaturePool(1,spec)
        try:
            future=pool.submit('file',str(path),service,48000,audio=True,waveform=True,need_metadata=True)
            deadline=time.monotonic()+15
            while not pool.features.full():
                self.assertLess(time.monotonic(),deadline)
                self.assertFalse(future.done())
                time.sleep(.01)
            started=time.monotonic();pool.cancel.set()
            result=future.result(timeout=2)
            self.assertTrue(result['cancelled'])
            self.assertLess(time.monotonic()-started,.5)
        finally:pool.close()
        self.assertEqual(list(service.waveform_work_dir.glob('index-audio-*.f32')),[])

    def test_real_process_features_metadata_waveform_and_queue_shutdown(self):
        e=spec_embedder();spec=FeatureSpec.from_embedder(e)
        service=AudioService(self.root/'cache');path=make_wave(self.root)
        expected=prepare_index_audio(service,path,48000,need_metadata=True)
        pool=FeaturePool(1,spec)
        try:
            future=pool.submit('file',str(path),service,48000,audio=True,waveform=True,need_metadata=True)
            features=[]
            deadline=time.monotonic()+15
            while not future.done() or len(features)<1:
                self.assertLess(time.monotonic(),deadline)
                features.extend(pool.read_batch(8))
            result=future.result()
            self.assertEqual(result['metadata'],expected.metadata)
            self.assertEqual(result['waveform'],expected.waveform)
            self.assertEqual(result['windows'],1)
            samples=next(expected.windows(480000))
            np.testing.assert_array_equal(features[0][2],window_features(samples,spec))
        finally:pool.close();expected.close()


class MemoryPool:
    """Deterministic bounded fake: real threads and cancellation, no models."""
    def __init__(self,workers,spec):
        self.cancel=threading.Event();self.features=queue.Queue(maxsize=32)
        self.pool=ThreadPoolExecutor(max_workers=workers);self.started=[];self.closed=False
        self.entered=threading.Event();self.release=None;self.slow=None;self.bad=None
    def submit(self,file_id,path,service,rate,*,audio,waveform,need_metadata):
        def work():
            n=int(Path(path).stem);self.started.append(n);self.entered.set()
            while self.release is not None and not self.release.is_set():
                if self.cancel.wait(.005):return dict(metadata=None,waveform=None,windows=0,cancelled=True)
            if n==self.slow:time.sleep(.15)
            error=ValueError('bad decode') if n==self.bad else None
            metadata=AudioMetadata(.2,48000,1,'wav') if need_metadata and not error else None
            count=0
            if audio and not error:
                for index in range(n%3+1):
                    while not self.cancel.is_set():
                        try:
                            self.features.put((file_id,index,np.array([[[[n+1.,index+1.,1.]]]],dtype=np.float32)),timeout=.01)
                            count+=1;break
                        except queue.Full:pass
            return dict(metadata=metadata,waveform=(WaveformResult([n/100]*2000,.2,48000,FileFingerprint(1,n))
                if waveform and not error else None),windows=count,decode_error=error,feature_error=None,cancelled=self.cancel.is_set())
        return self.pool.submit(work)
    def read_batch(self,size):
        out=[];deadline=time.monotonic()+.03
        while len(out)<size and not self.cancel.is_set():
            try:out.append(self.features.get(timeout=max(.001,deadline-time.monotonic())))
            except queue.Empty:break
        return out
    def close(self):self.cancel.set();self.pool.shutdown(wait=True);self.closed=True


class ContinuousTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='r5-pipeline-',dir=ROOT/'.test-data'))
        self.db=DatabaseManager(str(self.root/'db.sqlite'));self.ids=[];self.audio={};self.text={};self.batches=[];self.pool=None
        for n in range(40):
            p=make_wave(self.root,f'{n}.wav')
            self.ids.append(self.db.upsert_file(AudioFileRecord(path=str(p),filename=p.name)))
        self.e=spec_embedder()
        def forward(inputs,cancel):
            self.batches.append(inputs['input_features'].shape[0])
            return inputs['input_features'].numpy().reshape(-1,3)
        self.e._forward_audio_batch=forward
        def pack(path,meta,*,embedding,cancel):
            return dict(file_id=meta['file_id'],embedding=embedding.tolist(),metadata={'engine_fingerprint':'audio'})
        async def texts(records):
            return [dict(file_id=r['file_uuid'],embedding=[1.,0.],metadata={'engine_fingerprint':'text'}) for r in records]
        self.targets={'audio_vector':types.SimpleNamespace(prepare_audio=pack,write_prepared_audio=lambda rows:self.audio.update({r['file_id']:r for r in rows})),
                      'text_vector':types.SimpleNamespace(prepare_metadata=texts,write_prepared_metadata=lambda rows:self.text.update({r['file_id']:r for r in rows}))}
        self.job=self.db.create_job('default','index_pipeline_test');self.shutdown=False
        self.configure=lambda p:None
    def tearDown(self):self.db._get_connection().close()
    async def run_job(self):
        def factory(*args):
            self.pool=MemoryPool(*args);self.configure(self.pool);return self.pool
        with mock.patch.dict(os.environ,{'SOUNDBOT_PIPELINE_FILES':'32','SOUNDBOT_DECODE_PROCESSES':'4'}):
            return await resume_continuous(self.db,'default',self.job,['waveform','audio_vector','text_vector'],self.targets,
                AudioService(self.root/'cache'),embedder=self.e,run_audio=asyncio.to_thread,run_model=asyncio.to_thread,
                build_text=lambda r:dict(r),model_available=lambda:True,shutting_down=lambda:self.shutdown,pool_factory=factory)
    async def test_continuous_cross_file_batches_preserve_pooling_and_publish_while_next_decode_runs(self):
        self.configure=lambda p:setattr(p,'slow',0)
        published_starts=[];write=self.targets['audio_vector'].write_prepared_audio
        def writes(rows):published_starts.append(len(self.pool.started));write(rows)
        self.targets['audio_vector'].write_prepared_audio=writes
        failures,cancelled=await self.run_job()
        self.assertEqual(sum(failures.values()),0);self.assertFalse(cancelled)
        self.assertTrue(self.pool.closed);self.assertEqual(len(self.audio),40)
        self.assertTrue(any(n>=8 for n in self.batches))
        self.assertGreater(published_starts[0],16)
        for n,fid in enumerate(self.ids):
            windows=[np.array([n+1.,i+1.,1.],dtype=np.float32) for i in range(n%3+1)]
            expected=np.mean([v/np.linalg.norm(v) for v in windows],axis=0);expected/=np.linalg.norm(expected)
            np.testing.assert_allclose(self.audio[fid]['embedding'],expected,atol=1e-6)
        self.assertEqual(self.db.get_artifact_counts('default')['audio_vector']['ready'],40)
        self.assertEqual(self.db.get_job(self.job)['processed'],120)
        before=dict(self.audio);await self.run_job();self.assertEqual(self.audio,before)
    async def test_bad_file_model_and_chroma_failures_are_isolated(self):
        self.configure=lambda p:setattr(p,'bad',2)
        forward=self.e._forward_audio_batch
        def infer(inputs,cancel):
            if torch.any(inputs['input_features'][...,0]==4):raise ValueError('one bad model file')
            return forward(inputs,cancel)
        self.e._forward_audio_batch=infer
        write=self.targets['audio_vector'].write_prepared_audio
        def writes(rows):
            if any(r['file_id']==self.ids[5] for r in rows):raise ValueError('one bad Chroma row')
            write(rows)
        self.targets['audio_vector'].write_prepared_audio=writes
        failures,_=await self.run_job()
        self.assertEqual(failures,{'waveform':1,'audio_vector':3,'text_vector':1})
        self.assertEqual(len(self.audio),37);self.assertEqual(len(self.text),39)
        self.assertTrue(all(v['processing']==0 for v in self.db.get_artifact_counts('default').values()))

    async def test_text_batch_failure_isolated_and_ready_rows_not_recomputed(self):
        original=self.targets['text_vector'].prepare_metadata
        async def texts(records):
            if any(r['file_uuid']==self.ids[7] for r in records):raise ValueError('one bad text')
            return await original(records)
        self.targets['text_vector'].prepare_metadata=texts
        failures,_=await self.run_job()
        self.assertEqual(failures,{'waveform':0,'audio_vector':0,'text_vector':1})
        self.assertEqual(len(self.text),39)
        old_batches=len(self.batches)
        self.targets['text_vector'].prepare_metadata=original
        failures,_=await self.run_job()
        self.assertEqual(sum(failures.values()),0)
        self.assertEqual(len(self.batches),old_batches)
        self.assertEqual(len(self.text),40)

    async def test_claim_and_transport_bounds_and_disk_fault_leave_no_processing(self):
        def broken(*args,**kw):raise OSError('worker transport failed')
        self.configure=lambda p:setattr(p,'submit',broken)
        with self.assertRaisesRegex(OSError,'transport failed'):
            await asyncio.wait_for(self.run_job(),3)
        self.assertTrue(self.pool.closed)
        self.assertTrue(all(v['processing']==0 for v in self.db.get_artifact_counts('default').values()))
    async def test_cancel_request_and_shutdown_drain_and_restore_claims_under_half_second(self):
        for shutdown in (False,True):
            self.shutdown=False;self.job=self.db.create_job('default','index_cancel_test')
            self.configure=lambda p:setattr(p,'release',threading.Event())
            task=asyncio.create_task(self.run_job())
            while self.pool is None or not self.pool.entered.is_set():await asyncio.sleep(.005)
            started=time.monotonic()
            if shutdown:self.shutdown=True
            else:self.db.update_job(self.job,cancel_requested=1)
            _,cancelled=await asyncio.wait_for(task,3)
            self.assertTrue(cancelled);self.assertLess(time.monotonic()-started,.5)
            self.assertTrue(self.pool.closed);self.pool=None
            self.assertTrue(all(v['processing']==0 for v in self.db.get_artifact_counts('default').values()))
    async def test_cancel_watcher_database_failure_drains_and_propagates(self):
        original=self.db.get_job
        def read_job(job):
            if asyncio.current_task().get_name()=='index-cancel-watch':
                raise RuntimeError('watch database unavailable')
            return original(job)
        self.configure=lambda p:setattr(p,'release',threading.Event())
        with mock.patch.object(self.db,'get_job',side_effect=read_job):
            with self.assertRaisesRegex(RuntimeError,'watch database unavailable'):
                await asyncio.wait_for(self.run_job(),3)
        self.assertTrue(self.pool.closed)
        self.assertTrue(all(v['processing']==0 for v in self.db.get_artifact_counts('default').values()))
    async def test_worker_close_failure_still_shuts_writer_before_claim_restore(self):
        writers=[]
        class Writer(ThreadPoolExecutor):
            def shutdown(self,*args,**kw):
                super().shutdown(*args,**kw);writers.append(self)
        def configure(pool):
            original=pool.close
            def close():
                original();raise RuntimeError('pool close failed')
            pool.close=close
        self.configure=configure
        with mock.patch('core.index_pipeline.ThreadPoolExecutor',Writer):
            with self.assertRaisesRegex(RuntimeError,'pool close failed'):
                await asyncio.wait_for(self.run_job(),3)
        self.assertEqual(len(writers),1)
        self.assertTrue(all(not t.is_alive() for t in writers[0]._threads))
        self.assertTrue(all(v['processing']==0 for v in self.db.get_artifact_counts('default').values()))
    async def test_task_cancel_during_chroma_drains_before_restoring_claims(self):
        entered=threading.Event();release=threading.Event();exited=threading.Event()
        def write(rows):entered.set();release.wait(3);exited.set()
        self.targets['audio_vector'].write_prepared_audio=write
        task=asyncio.create_task(self.run_job())
        self.assertTrue(await asyncio.to_thread(entered.wait,3))
        task.cancel();await asyncio.sleep(.03)
        self.assertFalse(task.done());self.assertFalse(exited.is_set())
        release.set()
        with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,3)
        self.assertTrue(self.pool.closed);self.assertTrue(exited.is_set())
        self.assertTrue(all(v['processing']==0 for v in self.db.get_artifact_counts('default').values()))
        self.assertEqual(self.db.get_artifact_counts('default')['audio_vector']['ready'],0)
    async def test_fatal_sqlite_publication_restores_claims_and_crash_processing_recovers(self):
        with mock.patch('core.index_pipeline.publish_batch',side_effect=RuntimeError('fatal publish')):
            with self.assertRaisesRegex(RuntimeError,'fatal publish'):
                await asyncio.wait_for(self.run_job(),3)
        self.assertTrue(self.pool.closed)
        self.assertTrue(all(v['processing']==0 for v in self.db.get_artifact_counts('default').values()))
        self.db.set_artifact_state('default',self.ids[0],'audio_vector','processing')
        self.db._get_connection().close();self.db=DatabaseManager(str(self.root/'db.sqlite'))
        self.assertEqual(self.db.get_artifact('default',self.ids[0],'audio_vector')['state'],'pending')
