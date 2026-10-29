"""Isolated Windows/CUDA indexing benchmark; never connects to a running API.

Supply <=50 source paths in a JSON list. Audio is read in place. Outputs and
SQLite/Chroma are created only beneath this checkout's .test-data directory.
Run old and new checkouts sequentially, using the same samples/models/Python.
"""
def main():
    import argparse,asyncio,json,os,sys,time,tempfile,threading,subprocess,statistics
    from pathlib import Path
    from collections import defaultdict
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--samples',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--missing-metadata',action='store_true');p.add_argument('--warm-batches',action='store_true');a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];root=Path(tempfile.mkdtemp(prefix='task4-'+a.label+'-',dir=repo/'.test-data'))
    os.environ.update(SOUNDBOT_USER_DATA_DIR=str(root),SOUNDBOT_DATA_DIR=str(root),SOUNDBOT_MODELS_PATH=str(a.models.resolve()),PYTHONDONTWRITEBYTECODE='1',ENABLE_MODEL_PRELOAD='false')
    sys.path.insert(0,str(a.source.resolve()/'backend'))
    import numpy as np,torch,ctypes,types
    class Memory(ctypes.Structure):
     _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    def memory_info(pid=None):
     handle=kernel.OpenProcess(0x410,False,pid) if pid else ctypes.c_void_p(-1)
     info=Memory();info.cb=ctypes.sizeof(info)
     if handle:
      ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle) if pid else handle,ctypes.byref(info),info.cb)
     if pid and handle:kernel.CloseHandle(handle)
     return types.SimpleNamespace(rss=info.WorkingSetSize,peak=info.PeakWorkingSetSize)
    def process_cpu(pid):
     handle=kernel.OpenProcess(0x400,False,pid)
     if not handle:return 0.
     values=[ctypes.c_ulonglong() for _ in range(4)]
     kernel.GetProcessTimes(ctypes.c_void_p(handle),*[ctypes.byref(v) for v in values])
     kernel.CloseHandle(handle)
     return (values[2].value+values[3].value)/10_000_000
    pools=[];worker_cpu={};worker_rss={}
    def sample_workers():
     total=0
     for pool in pools:
      for child in (pool.pool._processes or {}).values():
       info=memory_info(child.pid);total+=info.rss
       worker_cpu[child.pid]=max(worker_cpu.get(child.pid,0),process_cpu(child.pid))
       worker_rss[child.pid]=max(worker_rss.get(child.pid,0),info.peak)
     return total
    
    # Bound the allocator to 3.1 GiB, leaving room for the CUDA context below 4 GiB.
    torch.cuda.set_per_process_memory_fraction(3.1*1024**3/torch.cuda.get_device_properties(0).total_memory)
    from core.embedder import CLIPEmbedder
    from core.audio_service import get_audio_service
    from core.database import DatabaseManager,AudioFileRecord
    from core.indexer import get_indexer,get_text_indexer
    import main
    from unittest import mock
    embedder=CLIPEmbedder();service=get_audio_service();db=DatabaseManager(str(root/'db'/'soundmind.db'))
    paths=json.loads(a.samples.read_text(encoding='utf-8'));ids=[]
    assert 0 < len(paths) <= 50, 'Sample limit is 50 files'
    for path in paths:
     meta=None if a.missing_metadata else service.probe(path);ids.append(db.upsert_file(AudioFileRecord(path=path,filename=Path(path).name,file_size=Path(path).stat().st_size,source_fingerprint=service.fingerprint(path).source_key,duration=meta.duration if meta else 0,sample_rate=meta.sample_rate if meta else 0,channels=meta.channels if meta else 0)))
    # Warm kernels once; excludes model loading and first-call initialization.
    embedder._process_audio_segment(np.zeros(48000,dtype=np.float32))
    if a.warm_batches:
     features=embedder._processor_inputs(np.zeros(48000,dtype=np.float32),cpu=True)
     with torch.inference_mode():
      for size in (1,2,4,8,16):
       inputs={key:value.repeat((size,)+(1,)*(value.ndim-1)) for key,value in features.items()}
       embedder._forward_audio_batch(inputs)
       text_inputs=embedder.processor(text=['sound effect '+('test '*100)]*size,return_tensors='pt',padding=True,truncation=True,max_length=embedder._text_max_length()).to(embedder.device)
       embedder.model.get_text_features(**text_inputs)
     torch.cuda.synchronize()
    if a.warm_batches:main._derive_ucs_metadata(['benchmark warmup'])
    metrics=defaultdict(float);calls=defaultdict(int);lock=threading.Lock();timeline=[];audio_batches=[];text_batches=[]
    per_file=defaultdict(lambda:defaultdict(float));context=threading.local()
    def wrap(obj,name,label,sync=False,path_index=None):
     orig=getattr(obj,name)
     def measured(*args,**kw):
      previous=getattr(context,'path',None)
      if path_index is not None:context.path=str(args[path_index])
      if sync:torch.cuda.synchronize()
      start=time.perf_counter()
      if label=='gpu_audio_forward':audio_batches.append(int(kw['input_features'].shape[0]))
      if label=='gpu_text_forward':text_batches.append(int(kw['input_ids'].shape[0]))
      try:return orig(*args,**kw)
      finally:
       if sync:torch.cuda.synchronize()
       elapsed=time.perf_counter()-start
       with lock:
        metrics[label]+=elapsed;calls[label]+=1;timeline.append((label,start,elapsed))
        if getattr(context,'path',None):per_file[context.path][label]+=elapsed
       context.path=previous
     setattr(obj,name,measured)
    wrap(main,'_build_text_metadata_payload','build_text')
    wrap(service,'waveform','waveform_total',path_index=0);wrap(service,'decode','clap_decode',path_index=0)
    try:
     from core import index_audio
     wrap(index_audio,'prepare_index_audio','shared_decode_wave')
     pool_init=index_audio.DecodePool.__init__;pool_close=index_audio.DecodePool.close
     def init_pool(self,*args,**kwargs):
      pool_init(self,*args,**kwargs);pools.append(self)
     def close_pool(self):
      sample_workers();return pool_close(self)
     index_audio.DecodePool.__init__=init_pool;index_audio.DecodePool.close=close_pool
     wrap(index_audio.DecodePool,'prepare','decode_worker_wait',path_index=2)
     wrap(embedder,'prepared_audio_to_embedding','prepared_embedding')
    except ImportError:pass
    try:
     from core.index_workers import FeaturePool
     init_feature_pool=FeaturePool.__init__;close_feature_pool=FeaturePool.close
     def init_features(self,*args,**kw):
      init_feature_pool(self,*args,**kw);pools.append(self)
     def close_features(self):
      sample_workers();return close_feature_pool(self)
     FeaturePool.__init__=init_features;FeaturePool.close=close_features
    except ImportError:pass
    wrap(embedder,'_processor_inputs','mel_and_transfer');wrap(embedder.model,'get_audio_features','gpu_audio_forward',True)
    wrap(embedder.model,'get_text_features','gpu_text_forward',True)
    from core import index_publish,index_resume
    orig=index_resume.publish_batch
    def publish(*args,**kw):
     start=time.perf_counter()
     try:return orig(*args,**kw)
     finally:metrics['sqlite_publish']+=time.perf_counter()-start
    index_resume.publish_batch=publish
    try:
     from core import index_pipeline
     index_pipeline.publish_batch=publish
    except ImportError:pass
    from core.indexer import AudioIndexer,TextMetadataIndexer
    wrap(AudioIndexer,'prepare_audio','audio_prepare_total',path_index=1)
    wrap(AudioIndexer,'write_prepared_audio','chroma_audio');wrap(TextMetadataIndexer,'write_prepared_metadata','chroma_text')
    process=types.SimpleNamespace(memory_info=memory_info,cpu_times=lambda:(time.process_time(),0));stop=threading.Event();rss=[];util=[]
    def monitor():
     while not stop.wait(.5):
      rss.append(process.memory_info().rss+sample_workers())
      try:
       result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,creationflags=0x08000000)
       util.append([float(x) for x in result.stdout.strip().split(',')])
      except Exception:pass
    thread=threading.Thread(target=monitor);thread.start();cpu0=sum(process.cpu_times()[:2]);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    job=db.create_job('default','index_task4_benchmark')
    try:
     with mock.patch.object(main,'get_db_manager',return_value=db):asyncio.run(main._repair_index_task_locked(job,'default',['waveform','audio_vector','text_vector'],'resume'))
     elapsed=time.perf_counter()-start;cpu=sum(process.cpu_times()[:2])-cpu0
    finally:stop.set();thread.join()
    jobdata=db.get_job(job);assert jobdata['state']=='completed',jobdata
    with db.get_cursor() as c: rows=c.execute('SELECT file_uuid,peaks_json FROM files').fetchall()
    indexer=get_indexer('default'); vectors=indexer.collection.get(ids=ids,include=['embeddings','metadatas'])
    metadata_byid=dict(zip(vectors['ids'],vectors['metadatas']));tv=get_text_indexer('default').collection.get(ids=ids,include=['embeddings','metadatas']);text_byid={fid:(v.tolist(),m) for fid,v,m in zip(tv['ids'],tv['embeddings'],tv['metadatas'])}
    with db.get_cursor() as c: dbmeta={r['file_uuid']:dict(r) for r in c.execute('SELECT file_uuid,duration,sample_rate,channels FROM files')}
    byid=dict(zip(vectors['ids'],vectors['embeddings'].tolist()));peaks=dict(rows)
    results=[{'path':path,'size':Path(path).stat().st_size,'peaks':json.loads(peaks[fid]),'vector':byid[fid],'audio_metadata':metadata_byid[fid],'text_vector':text_byid[fid][0],'text_metadata':text_byid[fid][1],'stream_metadata':dbmeta[fid]} for path,fid in zip(paths,ids)]
    report={'worker_profile':[pool.stats for pool in pools if hasattr(pool,'stats')],'warm_batches':a.warm_batches,'audio_batch_sizes':audio_batches,'text_batch_sizes':text_batches,'timeline':timeline,'label':a.label,'source':str(a.source),'fixture':str(root),'seconds':elapsed,'files_per_minute':len(paths)*60/elapsed,'MB_per_second':sum(Path(p).stat().st_size for p in paths)/1e6/elapsed,'cpu_cores_mean':(cpu+sum(worker_cpu.values()))/elapsed,'parent_cpu_cores_mean':cpu/elapsed,'worker_peak_rss':worker_rss,'peak_process_tree_rss_bytes':max(rss or [process.memory_info().rss]),'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved(),'gpu_global_util_mean':statistics.mean(x[0] for x in util) if util else None,'gpu_global_memory_peak_mib':max((x[1] for x in util),default=0),'profile_seconds':dict(metrics),'per_file_seconds':dict(per_file),'calls':dict(calls),'fingerprint':embedder.fingerprint,'job':jobdata,'results':results}
    output=repo/'.test-data'/('task4-'+a.label+'.json');output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','timeline','per_file_seconds')},ensure_ascii=False),flush=True)
    main._executor.shutdown(wait=True);main._model_executor.shutdown(wait=True);db._get_connection().close()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
