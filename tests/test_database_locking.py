"""Concurrent index writers must wait for SQLite's write lock, not fail at once."""
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'backend'))
from core.database import AudioFileRecord, DatabaseManager


class TransactionLockTests(unittest.TestCase):
    def setUp(self):
        # Thread-local connections stay open, so Windows cannot delete the file.
        (REPO_ROOT / '.test-data').mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='db-lock-', dir=REPO_ROOT / '.test-data'))
        self.db = DatabaseManager(str(root / 'lock.db'))
        audio = root / 'tone.wav'
        audio.write_bytes(b'RIFF')
        self.file_id = self.db.upsert_file(AudioFileRecord(path=str(audio), filename=audio.name), 'default')

    def test_read_then_write_publication_waits_for_a_claiming_writer(self):
        holding, release, errors = threading.Event(), threading.Event(), []

        def claim_and_hold():
            with self.db.transaction():
                self.db.claim_artifact_for_work('default', self.file_id, 'audio_vector')
                holding.set()
                release.wait(5)

        def publish_failure():
            try:
                # set_artifact_state reads the file row before it writes.
                with self.db.transaction():
                    self.db.set_artifact_state('default', self.file_id, 'waveform', 'failed',
                                               error_code='decode_failed', error_message='bad file')
            except sqlite3.Error as exc:
                errors.append(exc)

        claimer = threading.Thread(target=claim_and_hold)
        claimer.start()
        self.assertTrue(holding.wait(5))
        publisher = threading.Thread(target=publish_failure)
        publisher.start()
        publisher.join(.3)
        self.assertEqual(errors, [])  # The old deferred upgrade failed here at once.
        self.assertTrue(publisher.is_alive())
        release.set()
        claimer.join(5)
        publisher.join(10)
        self.assertEqual(errors, [])
        counts = self.db.get_artifact_counts('default')
        self.assertEqual(counts['audio_vector']['processing'], 1)
        self.assertEqual(counts['waveform']['failed'], 1)

    def test_failed_transaction_rolls_back_and_releases_the_lock(self):
        with self.assertRaises(RuntimeError):
            with self.db.transaction():
                self.db.claim_artifact_for_work('default', self.file_id, 'audio_vector')
                raise RuntimeError('publication failed')
        self.assertEqual(self.db.get_artifact_counts('default')['audio_vector']['pending'], 1)
        other = threading.Thread(target=lambda: self.assertTrue(
            self.db.claim_artifact_for_work('default', self.file_id, 'audio_vector')))
        other.start()
        other.join(5)
        self.assertEqual(self.db.get_artifact_counts('default')['audio_vector']['processing'], 1)

    def test_nested_transactions_commit_once_at_the_outer_level(self):
        with self.db.transaction():
            self.db.claim_artifact_for_work('default', self.file_id, 'audio_vector')
            with self.assertRaises(ValueError):
                with self.db.transaction():
                    self.db.claim_artifact_for_work('default', self.file_id, 'text_vector')
                    raise ValueError('inner failure')
        counts = self.db.get_artifact_counts('default')
        self.assertEqual(counts['audio_vector']['processing'], 1)
        self.assertEqual(counts['text_vector']['pending'], 1)


if __name__ == '__main__':
    unittest.main()
