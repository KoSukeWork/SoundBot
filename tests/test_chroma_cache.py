"""Regression for Chroma 1.5.5's unflushed HNSW cache eviction."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from core.chroma_client import persistent_client


class ChromaCacheTests(unittest.TestCase):
    def test_small_and_persisted_collections_keep_vectors_during_other_writes(self):
        root = Path(tempfile.mkdtemp(prefix='chroma-cache-', dir=ROOT / '.test-data'))
        client = persistent_client(root)
        try:
            # 100 indexes exceed the original 64 one-slot shards. The old
            # implementation deterministically loses at least 36 indexes.
            collections = []
            for i in range(100):
                col = client.create_collection(f'cache-{i}', embedding_function=None)
                col.count()  # Populate the executor's backfilled set first.
                if i == 0:
                    col.upsert(ids=[str(j) for j in range(1000)],
                               embeddings=[[1., 0., 0.]] * 1000)
                col.upsert(ids=['tail'], embeddings=[[0., 1., 0.]],
                           metadatas=[{'role': 'tail'}])
                collections.append(col)
            def read_all():
                for col in collections:
                    self.assertEqual(col.get(ids=['tail'], include=['embeddings'])
                                     ['embeddings'].tolist(), [[0., 1., 0.]])
                    self.assertEqual(col.query(query_embeddings=[[0., 1., 0.]],
                                               where={'role': 'tail'}, n_results=1)['ids'], [['tail']])
            with ThreadPoolExecutor(max_workers=2) as pool:
                reader = pool.submit(read_all)
                for col in collections:
                    col.upsert(ids=['next'], embeddings=[[0., 0., 1.]])
                reader.result()
            read_all()
            for col in collections:
                self.assertEqual(col.get(ids=['next'], include=['embeddings'])
                                 ['embeddings'].tolist(), [[0., 0., 1.]])
        finally:
            client.close()
