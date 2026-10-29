from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from core.indexer import AudioIndexer, TextMetadataIndexer


class EnginePublicationTests(unittest.TestCase):
    def test_prepare_cannot_observe_first_upsert_before_engine_cache_publication(self):
        for cls, method in ((AudioIndexer,'write_prepared_audio'),(TextMetadataIndexer,'write_prepared_metadata')):
            with self.subTest(indexer=cls.__name__):
                indexer=object.__new__(cls)
                indexer._write_lock=threading.RLock()
                indexer._engine_fingerprints=set();indexer._unknown_engine_metadata=False
                indexer.needs_rebuild=False;indexer.persist_directory='isolated';indexer.collection_name='test'
                stored=threading.Event();release=threading.Event();started=threading.Event();finished=threading.Event()
                def upsert(**kwargs):
                    stored.set();self.assertTrue(release.wait(3))
                indexer.collection=types.SimpleNamespace(count=lambda:int(stored.is_set()),upsert=upsert)
                def prepare():
                    started.set()
                    try:indexer._ensure_engine_compatible('engine')
                    finally:finished.set()
                with mock.patch('core.indexer.bump_collection_revision'),ThreadPoolExecutor(2) as pool:
                    writer=pool.submit(getattr(indexer,method),[dict(file_id='a',embedding=[1.,0.],metadata={'engine_fingerprint':'engine'})])
                    try:
                        self.assertTrue(stored.wait(2));reader=pool.submit(prepare)
                        self.assertTrue(started.wait(2))
                        self.assertFalse(finished.wait(.05))
                    finally:release.set()
                    writer.result();reader.result()
                self.assertFalse(indexer.needs_rebuild)
                self.assertEqual(indexer._engine_fingerprints,{'engine'})
                with self.assertRaisesRegex(RuntimeError,'fingerprint changed'):
                    indexer._ensure_engine_compatible('different-engine')
