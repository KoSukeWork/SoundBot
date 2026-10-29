"""Isolated end-to-end migration + real backend restart; never targets user data.

The core's machine-wide offline check is mocked only inside this synthetic
test so an unrelated live SoundBot can continue running. Production CLI has
no bypass. All filesystem operations remain below this script's scratch root.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from unittest import mock
import urllib.request
import wave

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0,str(REPO/'backend'))


def snapshot(database):
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as connection:
        return connection.execute('SELECT file_uuid,kind,state,source_fingerprint,engine_fingerprint,updated_at FROM file_artifacts ORDER BY file_uuid,kind').fetchall()


def seed(root):
    from core.database import DatabaseManager,AudioFileRecord
    from core.audio_service import AudioService
    from core.indexer import get_chroma_client
    user=root/'profile'
    db=DatabaseManager(str(user/'db'/'soundmind.db'))
    service=AudioService(root/'wave-cache')
    client=get_chroma_client(str(user/'chroma_projects'/'default'))
    audio=client.get_or_create_collection('audio_embeddings',embedding_function=None,metadata={'hnsw:space':'cosine'})
    text=client.get_or_create_collection('text_metadata_embeddings',embedding_function=None,metadata={'hnsw:space':'cosine'})
    for i in range(3):
        path=root/f'tone-{i}.wav'
        with wave.open(str(path),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(8000)
            wav.writeframes(b'\x00\x10'*800)
        fp=service.fingerprint(path)
        file_id=db.upsert_file(AudioFileRecord(path=str(path),filename=path.name,duration=.1,sample_rate=8000,channels=1,source_fingerprint=fp.source_key))
        for kind,collection in [('audio_vector',audio),('text_vector',text)]:
            collection.upsert(ids=[file_id],embeddings=[[1.,0.]],metadatas=[{'engine_fingerprint':'fixture','duration':.1,'file_path':str(path)}])
            db.set_artifact_state('default',file_id,kind,'ready',engine_fingerprint='fixture')
        if i<2:
            result=service.waveform(path)
            db.update_peaks(str(path),result.peaks,'default',source_fingerprint=fp.source_key,waveform_fingerprint=fp.key)
    (root/'before.json').write_text(json.dumps(snapshot(user/'db'/'soundmind.db')))
    (user/'user_config.json').write_text(json.dumps({'tempClipDir':str(root/'clips'),'keep':'profile-settings'}))
    db._get_connection().close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed-root',type=Path)
    args=parser.parse_args()
    if args.seed_root:
        root=args.seed_root.resolve();root.relative_to(REPO/'.test-data')
        seed(root);return
    root=Path(tempfile.mkdtemp(prefix='data-migration-e2e-',dir=REPO/'.test-data'))
    user,target=root/'profile',root/'new-data'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',SOUNDBOT_USER_DATA_DIR=str(user),
             SOUNDBOT_MODELS_PATH=str(root/'no-models'),ENABLE_MODEL_PRELOAD='false',
             SOUNDBOT_CONTROL_TOKEN=secrets.token_hex(24),SOUNDBOT_START_PAUSED='0',PYTHONIOENCODING='utf-8')
    env.pop('SOUNDBOT_DATA_DIR',None)
    with (root/'seed.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--seed-root',str(root)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    from core.data_migration import migrate
    with mock.patch.dict(os.environ,env,clear=True), mock.patch('core.data_migration.require_backend_offline'):
        migration=migrate(user,target)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env['SOUNDBOT_PORT']=str(port)
    log=(root/'backend.log').open('w',encoding='utf-8')
    child=subprocess.Popen([sys.executable,str(REPO/'backend'/'main.py')],env=env,cwd=REPO/'backend',stdout=log,stderr=subprocess.STDOUT)
    def request(route,method='GET'):
        req=urllib.request.Request(f'http://127.0.0.1:{port}/api/v1/{route}',method=method,headers={'X-SoundBot-Control':env['SOUNDBOT_CONTROL_TOKEN']})
        with urllib.request.urlopen(req,timeout=3) as response:return json.load(response)
    try:
        deadline=time.monotonic()+75
        status=None
        while time.monotonic()<deadline:
            if child.poll() is not None: raise RuntimeError(f'Test backend exited: {child.returncode}')
            try:
                status=request('projects/default/index/status')
                if status.get('job',{}).get('state')=='completed':break
            except (OSError,ValueError):pass
            time.sleep(.25)
        else: raise RuntimeError('Test backend did not finish recovery')
        assert status['job']['processed']==status['job']['total']==1,status
        request('system/shutdown','POST')
        child.wait(timeout=30)
        assert child.returncode==0
    finally:
        if child.poll() is None:
            # Only this script's own synthetic backend may be terminated on test failure.
            child.terminate();child.wait(timeout=15)
        log.close()
    before=json.loads((root/'before.json').read_text())
    after=snapshot(target/'db'/'soundmind.db')
    ready_before={tuple(row[:2]):tuple(row) for row in before if row[2]=='ready'}
    ready_after={tuple(row[:2]):tuple(row) for row in after}
    assert all(ready_after[key]==row for key,row in ready_before.items())
    assert all(row[2]=='ready' for row in after)
    from core.indexer import get_chroma_client
    client=get_chroma_client(str(target/'chroma_projects'/'default'))
    for name in ('audio_embeddings','text_metadata_embeddings'):
        result=client.get_collection(name,embedding_function=None).get(include=['embeddings'])
        assert len(result['ids'])==3
        assert result['embeddings'].tolist()==[[1.,0.]]*3
    report={'root':str(root),'migration':migration,'resumed_job':status['job'],
            'ready_rows_preserved':len(ready_before),'ready_after':len(after),
            'chroma_vectors_preserved':6,'backend_exit_code':child.returncode}
    (REPO/'.test-data'/'task3-e2e.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
