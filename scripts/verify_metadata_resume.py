"""Real CLAP/SQLite/Chroma check using generated stereo audio and isolated data."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import wave
from unittest import mock


def main():
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parents[1]
    data = Path(tempfile.mkdtemp(prefix='metadata-real-', dir=root / '.test-data'))
    os.environ['SOUNDBOT_USER_DATA_DIR'] = str(data)
    # Set SOUNDBOT_MODELS_PATH before invoking this script if necessary.
    sys.path.insert(0,str(root / 'backend'))
    import torch
    import numpy as np
    torch.set_num_threads(2)
    from core.database import DatabaseManager, AudioFileRecord
    from core.audio_service import get_audio_service
    from core.embedder import get_embedder
    from core.index_lifecycle import get_active_audio_indexer, get_active_text_indexer
    import main as backend
    db = DatabaseManager(str(data / 'db' / 'soundmind.db'))
    embedder = get_embedder()
    assert embedder is not None
    ids = []
    for i in range(3):
        path = data / f'stereo-{i}.wav'
        samples = (np.sin(np.arange(24000) * (2*np.pi*(220+i*100)/96000))*12000).astype('<i2')
        with wave.open(str(path),'wb') as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(96000)
            wav.writeframes(np.repeat(samples[:,None],2,axis=1).tobytes())
        ids.append(db.upsert_file(AudioFileRecord(path=str(path),filename=path.name,
            source_fingerprint=get_audio_service().fingerprint(path).source_key)))
    job = db.create_job('default','index_auto_metadata_real')
    with mock.patch.object(backend,'get_db_manager',return_value=db):
        asyncio.run(backend._repair_index_task_locked(job,'default',['waveform','audio_vector','text_vector'],'resume'))
    status = db.get_job(job)
    assert status['state']=='completed',status
    audio, text = get_active_audio_indexer(db,'default'), get_active_text_indexer(db,'default')
    for file_id in ids:
        record = db.get_file_by_uuid(file_id,'default')
        assert (record.duration,record.sample_rate,record.channels)==(.25,96000,2)
        for indexer in (audio,text):
            result = indexer.collection.get(ids=[file_id],include=['metadatas','embeddings'])
            metadata = result['metadatas'][0]
            assert (metadata['duration'],metadata['sample_rate'],metadata['channels'])==(.25,96000,2)
            assert np.isfinite(result['embeddings']).all()
    result = {'data':str(data),'torch':torch.__version__,'device':str(embedder.device),
              'job':status,'counts':db.get_artifact_counts('default'),
              'verified_metadata':{'duration':.25,'sample_rate':96000,'channels':2},
              'fingerprint':embedder.fingerprint}
    (root / '.test-data' / 'task2-real-clap.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
    db._get_connection().close()


if __name__=='__main__':
    main()
