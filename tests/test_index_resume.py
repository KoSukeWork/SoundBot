from __future__ import annotations

import asyncio
from collections import Counter
from itertools import count
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
_DATA = tempfile.TemporaryDirectory()
os.environ.setdefault("SOUNDBOT_USER_DATA_DIR", _DATA.name)

import main
from core.database import DatabaseManager, canonicalize_path
from core.audio_service import FileFingerprint, WaveformResult


def seed(manager, root, count):
    canonical_root = canonicalize_path(str(root))
    with manager.get_cursor() as cursor:
        cursor.executemany("""
            INSERT INTO files(file_uuid, path, canonical_path, filename, project_id, peaks_json,
                              duration, sample_rate, channels)
            VALUES (?, ?, ?, ?, 'default', '[0.1, 0.2]', 1, 48000, 1)
        """, ((str(i), str(root / f'{i}.wav'), f'{canonical_root}/{i}.wav', f'{i}.wav')
              for i in range(count)))
        cursor.execute("""
            INSERT INTO file_artifacts(project_id, file_uuid, kind)
            SELECT project_id, file_uuid, 'waveform' FROM files
        """)


class ResumeDatabaseTests(unittest.TestCase):
    def test_paging_crosses_100000_without_revisiting_failures_or_loading_peaks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = DatabaseManager(str(root / "resume.db"))
            try:
                seed(manager, root, 100005)
                manager.set_artifact_state("default", "0", "waveform", "ready")
                manager.set_artifact_state("default", "1", "waveform", "failed")
                self.assertEqual(manager.count_artifacts_for_work("default"), 100004)
                seen = set()
                for row in manager.iter_artifacts_for_work("default", batch_size=127):
                    self.assertNotIn("peaks_json", row)
                    self.assertNotIn(row["file_uuid"], seen)
                    seen.add(row["file_uuid"])
                self.assertEqual(len(seen), 100004)
                self.assertIn("100004", seen)
                self.assertIn("1", seen)
                self.assertNotIn("0", seen)
            finally:
                manager._get_connection().close()


class ResumeJobTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manager = DatabaseManager(str(self.root / "resume.db"))
        seed(self.manager, self.root, 7)
        main._shutting_down = False
        main._automatic_index_tasks.clear()
        self.calls = Counter()

    def tearDown(self):
        self.manager._get_connection().close()
        self.temp.cleanup()

    async def run_waveforms(self, *, fail=None, cancel_after=None):
        job_id = self.manager.create_job("default", "index_auto_test")

        def waveform(path):
            file_id = Path(path).stem
            self.calls[file_id] += 1
            if cancel_after and sum(self.calls.values()) == cancel_after:
                self.manager.update_job(job_id, cancel_requested=1)
            if file_id == fail:
                raise RuntimeError("permanent bad file")
            return WaveformResult([0.1] * 2000, 1.0, 48000, FileFingerprint(10, 123))

        async def run_inline(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        original = self.manager.iter_artifacts_for_work
        with (
            mock.patch.object(main, "get_db_manager", return_value=self.manager),
            mock.patch("core.audio_service.get_audio_service", return_value=mock.Mock(waveform=waveform)),
            mock.patch.object(main, "_run_audio_work", side_effect=run_inline),
            mock.patch('core.index_resume.time', mock.Mock(monotonic=mock.Mock(side_effect=count()))),
            mock.patch.object(self.manager, "get_files_by_project", side_effect=AssertionError("full records")),
            mock.patch.object(self.manager, "get_vector_reconcile_records", side_effect=AssertionError("deep repair")),
            mock.patch.object(self.manager, "iter_artifacts_for_work", side_effect=lambda *a: original(*a, batch_size=2)),
            mock.patch("core.search_engine.reset_optimized_searcher"),
            mock.patch("core.ai_chat_service.reset_ai_chat_service"),
        ):
            await main._repair_index_task_locked(job_id, "default", ["waveform"], "resume")
        return self.manager.get_job(job_id)

    async def test_pages_finish_and_failed_rows_are_attempted_once(self):
        self.manager.set_artifact_state("default", "0", "waveform", "ready")
        before = self.manager.get_artifact("default", "0", "waveform")
        self.manager.set_artifact_state("default", "1", "waveform", "failed")
        self.manager.set_artifact_state("default", "2", "waveform", "stale")
        job = await self.run_waveforms(fail="1")
        self.assertEqual(job["state"], "failed")
        self.assertEqual(job["stage"], "done_with_errors")
        self.assertEqual((job["processed"], job["total"]), (6, 6))
        self.assertEqual(self.calls, Counter({str(i): 1 for i in range(1, 7)}))
        self.assertEqual(before, self.manager.get_artifact("default", "0", "waveform"))
        failed = self.manager.get_artifact("default", "1", "waveform")
        self.assertEqual(failed["state"], "failed")
        self.assertIn("permanent bad file", failed["error_message"])
        self.assertEqual(self.manager.get_artifact_counts("default")["waveform"]["ready"], 6)

    async def test_cancel_and_restart_only_process_unfinished_rows(self):
        job = await self.run_waveforms(cancel_after=3)
        self.assertEqual((job["state"], job["processed"]), ("cancelled", 3))
        self.manager.set_artifact_state("default", "3", "waveform", "processing")
        self.manager._get_connection().close()
        self.manager = DatabaseManager(str(self.root / "resume.db"))
        self.assertEqual(self.manager.get_artifact("default", "3", "waveform")["state"], "pending")
        resumed = await self.run_waveforms()
        self.assertEqual((resumed["state"], resumed["processed"], resumed["total"]), ("completed", 4, 4))
        self.assertEqual(self.calls, Counter({str(i): 1 for i in range(7)}))

    async def test_missing_model_still_schedules_waveform_and_ui_exposes_job(self):
        repair = mock.AsyncMock()
        with mock.patch.object(main, "get_db_manager", return_value=self.manager), mock.patch.object(main, "_repair_index_task", repair):
            main._schedule_waveform_reconcile()
            payload = await main.get_project_index_status("default")
            self.assertEqual(payload["job"]["total"], 7)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        self.assertEqual(repair.await_args.args[2:], (["waveform"], "resume"))

    async def test_failed_text_is_retried_without_reconciling_ready_vectors(self):
        self.manager.set_artifact_state("default", "0", "text_vector", "failed", error_code="text_index_failed")
        self.manager.set_artifact_state("default", "1", "text_vector", "ready", engine_fingerprint="old-compatible")
        before = self.manager.get_artifact("default", "1", "text_vector")
        indexer = mock.Mock()
        indexer.get_manifest.return_value = {
            "collection": "text_metadata_embeddings", "metric": "cosine",
            "needs_rebuild": False, "engine_fingerprint": "old-compatible",
        }
        indexer.prepare_metadata = mock.AsyncMock(return_value=[{
            "file_id": "0", "embedding": [1., 0.], "metadata": {"engine_fingerprint": "old-compatible"},
        }])
        job_id = self.manager.create_job("default", "index_auto_test")
        with (
            mock.patch.object(main, "get_db_manager", return_value=self.manager),
            mock.patch.object(main, "_get_active_text_indexer", return_value=indexer),
            mock.patch("core.search_engine.reset_optimized_searcher"),
            mock.patch("core.ai_chat_service.reset_ai_chat_service"),
        ):
            await main._repair_index_task_locked(job_id, "default", ["text_vector"], "resume")
        indexer.reconcile.assert_not_called()
        indexer.prepare_metadata.assert_awaited_once()
        indexer.write_prepared_metadata.assert_called_once()
        self.assertEqual(self.manager.get_artifact("default", "0", "text_vector")["state"], "ready")
        self.assertEqual(self.manager.get_artifact("default", "1", "text_vector"), before)
        job = self.manager.get_job(job_id)
        self.assertEqual((job["state"], job["processed"], job["total"]), ("completed", 1, 1))

    async def test_model_ready_event_during_waveform_job_is_not_lost(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def repair(*args):
            if args[2] == ["waveform"]:
                entered.set()
                await release.wait()

        with mock.patch.object(main, "get_db_manager", return_value=self.manager), mock.patch.object(main, "_repair_index_task", side_effect=repair) as worker:
            main._schedule_waveform_reconcile()
            await entered.wait()
            # Add pending vector artifacts, as in a registered import.
            with self.manager.get_cursor() as cursor:
                cursor.execute("INSERT INTO file_artifacts(project_id, file_uuid, kind) VALUES ('default', '0', 'text_vector')")
            main._schedule_model_ready_reconcile()
            release.set()
            for _ in range(5):
                await asyncio.sleep(0)
        self.assertEqual(worker.await_count, 2)
        self.assertIn("text_vector", worker.await_args.args[2])


if __name__ == "__main__":
    unittest.main()
