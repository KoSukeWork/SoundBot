"""Synthetic storage benchmark; never opens a user library or loads a model.

Run seed once, then run baseline/current against the same isolated fixture.
Embeddings and waveforms are deterministic stand-ins: timings measure storage
and scheduling overhead, NOT achievable end-to-end GPU throughput.
"""
import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from unittest import mock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--seed', action='store_true')
    parser.add_argument('--files', type=int, default=800000)
    parser.add_argument('--sample', type=int, default=512)
    parser.add_argument('--label', default='current')
    parser.add_argument('--prime-vectors', type=int, default=0,
                        help='Prepopulate each isolated Chroma collection, then exit')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    data = args.data.resolve()
    # This script is deliberately restricted to scratch data below this repo.
    data.relative_to(Path(__file__).resolve().parents[1] / '.test-data')
    os.environ['SOUNDBOT_USER_DATA_DIR'] = str(data)
    sys.path.insert(0, str(args.backend.resolve()))
    from core.database import DatabaseManager, canonicalize_path
    from core.audio_service import FileFingerprint, WaveformResult
    from core.indexer import AudioIndexer, TextMetadataIndexer
    import main as backend
    import numpy as np

    db = DatabaseManager(str(data / 'db' / 'soundmind.db'))
    peaks = [0.1] * 2000
    if args.seed:
        if db.get_file_count():
            raise RuntimeError('Seed requires an empty synthetic database')
        start = time.perf_counter()
        with db.get_cursor() as cur:
            for offset in range(0, args.files, 5000):
                cur.executemany('''INSERT INTO files
                    (file_uuid,path,canonical_path,filename,project_id,duration,sample_rate,channels,peaks_json,source_fingerprint)
                    VALUES (?,?,?,?, 'default',1,48000,1,?,?)''',
                    ((str(i), str(data / f'{i}.wav'), canonicalize_path(str(data / f'{i}.wav')), f'{i}.wav',
                      json.dumps(peaks) if i % 20 == 0 else None, FileFingerprint(10,123).source_key)
                     for i in range(offset, min(offset+5000, args.files))))
            for kind in ('waveform', 'audio_vector', 'text_vector'):
                cur.execute('''INSERT INTO file_artifacts(project_id,file_uuid,kind,state)
                    SELECT project_id,file_uuid,?,'ready' FROM files''', (kind,))
        print(json.dumps({'seed_seconds': time.perf_counter()-start,
                          'files': args.files, 'artifacts': args.files*3,
                          'peaks_rows': (args.files+19)//20,
                          'bytes': Path(db.db_path).stat().st_size}), flush=True)
        db._get_connection().close()
        return

    with db.get_cursor() as cur:
        cur.execute("UPDATE file_artifacts SET state='pending' WHERE file_uuid IN (SELECT file_uuid FROM files ORDER BY id LIMIT ?)", (args.sample,))
    vector = np.ones(512, dtype=np.float32) / np.sqrt(512)
    class Provider:
        fingerprint = 'benchmark'
        async def embed_texts(self, texts):
            return [vector for _ in texts]
    audio = AudioIndexer(str(data / 'chroma_projects' / 'default'), project_id='default')
    text = TextMetadataIndexer(str(data / 'chroma_projects' / 'default'), project_id='default', provider=Provider())
    if args.prime_vectors:
        rng = np.random.default_rng(42)
        for offset in range(args.sample, args.prime_vectors, 1000):
            ids = [str(i) for i in range(offset,min(offset+1000,args.prime_vectors))]
            vectors = rng.standard_normal((len(ids),512)).astype(np.float32)
            vectors /= np.linalg.norm(vectors,axis=1,keepdims=True)
            for indexer in (audio,text):
                indexer.collection.upsert(ids=ids,embeddings=vectors,
                    metadatas=[{'engine_fingerprint':'benchmark','duration':1.0} for _ in ids])
        print(json.dumps({'primed_audio':audio.collection.count(),'primed_text':text.collection.count()}),flush=True)
        db._get_connection().close()
        return
    def generate(file_id, path, info, embedder):
        return {'file_id': file_id, 'embedding': vector.tolist(), 'metadata': {
            'filename': info.filename, 'duration': info.duration, 'sample_rate': info.sample_rate,
            'channels': info.channels, 'file_path': path, 'engine_fingerprint': 'benchmark'}}
    async def inline(fn, *a, **kw):
        return fn(*a, **kw)
    service = mock.Mock(waveform=lambda path: WaveformResult(peaks,1,48000,FileFingerprint(10,123)))
    job_id = db.create_job('default', 'index_auto_benchmark')
    transactions = []
    chroma_writes = Counter()
    original_audio_upsert, original_text_upsert = audio.collection.upsert, text.collection.upsert
    def write_audio(**kw):
        chroma_writes['audio'] += 1
        return original_audio_upsert(**kw)
    def write_text(**kw):
        chroma_writes['text'] += 1
        return original_text_upsert(**kw)
    # Nested savepoint RELEASE does not commit to disk.
    db._get_connection().set_trace_callback(lambda sql: transactions.append(sql.split()[0])
                                            if sql.strip().upper() in ('COMMIT','RELEASE SOUNDBOT_BATCH_0') else None)
    with (mock.patch.object(backend,'get_db_manager',return_value=db),
          mock.patch.object(backend,'_get_active_audio_indexer',return_value=audio),
          mock.patch.object(backend,'_get_active_text_indexer',return_value=text),
          mock.patch.object(backend,'is_embedder_available',return_value=True),
          mock.patch.object(backend,'_derive_ucs_metadata',return_value={'search_text':'','categories':[],'keywords':[]}),
          mock.patch.object(backend,'_run_audio_work',side_effect=inline),
          mock.patch.object(backend,'_run_model_work',side_effect=inline),
          mock.patch('core.indexer.get_embedder',return_value=mock.Mock(fingerprint='benchmark')),
          mock.patch('core.indexer.os.path.getsize',return_value=10),
          mock.patch.object(audio,'_generate_embedding_for_file',side_effect=generate),
          mock.patch.object(audio.collection,'upsert',side_effect=write_audio),
          mock.patch.object(text.collection,'upsert',side_effect=write_text),
          mock.patch('core.audio_service.get_audio_service',return_value=service),
          mock.patch.object(audio,'get_manifest',return_value={'needs_rebuild':False,'engine_fingerprint':'benchmark','collection':'audio_embeddings'}),
          mock.patch.object(text,'get_manifest',return_value={'needs_rebuild':False,'engine_fingerprint':'benchmark','collection':'text_metadata_embeddings'})):
        start=time.perf_counter()
        asyncio.run(backend._repair_index_task_locked(job_id,'default',['waveform','audio_vector','text_vector'],'resume'))
        elapsed=time.perf_counter()-start
    job=db.get_job(job_id)
    assert job['state']=='completed', job
    result={'label':args.label,'sample_files':args.sample,'seconds':elapsed,
            'files_per_second':args.sample/elapsed,'sqlite_commits':len(transactions),
            'chroma_upserts':dict(chroma_writes),
            'collection_sizes':{'audio':audio.collection.count(),'text':text.collection.count()},
            'job':job,'storage_only':True}
    (data / f'{args.label}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    db._get_connection().close()


if __name__=='__main__':
    main()
