"""Engine fingerprint checks must stay exact without scanning large collections."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from core.chroma_client import persistent_client
from core.indexer import bump_collection_revision, collection_engine_fingerprints


class CountingCollection:
    def __init__(self, collection):
        self.collection = collection
        self.id = collection.id
        self.gets = []

    def count(self):
        return self.collection.count()

    def get(self, **kwargs):
        self.gets.append(kwargs)
        return self.collection.get(**kwargs)


class EngineFingerprintTests(unittest.TestCase):
    def setUp(self):
        (ROOT / '.test-data').mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='engine-fp-', dir=ROOT / '.test-data'))
        self.client = persistent_client(root)
        self.addCleanup(self.client.close)

    def collection(self, name, metadatas):
        col = self.client.create_collection(name, embedding_function=None)
        for start in range(0, len(metadatas), 1000):
            chunk = metadatas[start:start + 1000]
            col.add(ids=[str(start + i) for i in range(len(chunk))],
                    embeddings=[[1., 0.]] * len(chunk), metadatas=chunk)
        return CountingCollection(col)

    def test_single_engine_uses_two_probes_instead_of_paging(self):
        col = self.collection('single', [{'engine_fingerprint': 'clap:a', 'n': i} for i in range(2500)])
        self.assertEqual(collection_engine_fingerprints(col), ({'clap:a'}, False))
        self.assertEqual(len(col.gets), 2)
        self.assertTrue(all(get.get('limit') == 1 for get in col.gets))

    def test_finds_second_engine_beyond_the_first_page(self):
        metadatas = [{'engine_fingerprint': 'clap:a'}] * 3500 + [{'engine_fingerprint': 'clap:b'}]
        col = self.collection('mixed', metadatas)
        self.assertEqual(collection_engine_fingerprints(col), ({'clap:a', 'clap:b'}, False))
        self.assertEqual([get.get('limit') for get in col.gets], [1, 1, 1])

    def test_blank_fingerprint_is_unknown(self):
        col = self.collection('blank', [{'engine_fingerprint': 'clap:a'}] * 1500 + [{'engine_fingerprint': ' '}])
        self.assertEqual(collection_engine_fingerprints(col), ({'clap:a'}, True))

    def test_missing_key_falls_back_to_exact_scan(self):
        metadatas = ([{'engine_fingerprint': 'clap:a'}] * 1200 + [{'legacy': True}]
                     + [{'engine_fingerprint': 'clap:b'}])
        col = self.collection('missing', metadatas)
        self.assertEqual(collection_engine_fingerprints(col), ({'clap:a', 'clap:b'}, True))

    def test_repeated_checks_reuse_the_answer_until_a_write(self):
        col = self.collection('cached', [{'engine_fingerprint': 'clap:a'}] * 10)
        self.assertEqual(collection_engine_fingerprints(col), ({'clap:a'}, False))
        probes = len(col.gets)
        self.assertEqual(collection_engine_fingerprints(col), ({'clap:a'}, False))
        self.assertEqual(len(col.gets), probes)
        # An indexer write bumps the revision even when the count is unchanged.
        col.collection.upsert(ids=['0'], embeddings=[[1., 0.]], metadatas=[{'engine_fingerprint': 'clap:b'}])
        bump_collection_revision('unused-dir', 'cached')
        self.assertEqual(collection_engine_fingerprints(col), ({'clap:a', 'clap:b'}, False))
        # A count change is also never served from the cache.
        col.collection.add(ids=['new'], embeddings=[[1., 0.]], metadatas=[{'legacy': True}])
        self.assertEqual(collection_engine_fingerprints(col), ({'clap:a', 'clap:b'}, True))

    def test_empty_collection(self):
        col = CountingCollection(self.client.create_collection('empty', embedding_function=None))
        self.assertEqual(collection_engine_fingerprints(col), (set(), False))
        self.assertEqual(col.gets, [])


if __name__ == '__main__':
    unittest.main()
