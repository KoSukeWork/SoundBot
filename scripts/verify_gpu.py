"""Run real CUDA/CPU inference against disposable fixtures; never open app data."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import wave


def main():
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scratch = root / ".test-data"
    scratch.mkdir(exist_ok=True)
    # Retain synthetic fixtures/logs for inspection. SQLite connections owned by
    # worker threads may keep Windows handles until the verification process exits.
    with tempfile.TemporaryDirectory(prefix="gpu-", dir=scratch, delete=False) as directory:
        os.environ["SOUNDBOT_USER_DATA_DIR"] = directory
        os.environ["SOUNDBOT_DATA_DIR"] = directory
        os.environ["SOUNDBOT_MODELS_PATH"] = str(args.models.resolve())
        os.environ["SOUNDBOT_PORT"] = "8011"
        sys.path.insert(0, str(root / "backend"))
        import numpy as np
        import torch
        from core.embedder import CLIPEmbedder, reset_embedder
        from core.audio_service import get_audio_service
        from core.database import AudioFileRecord, DatabaseManager

        torch.set_num_threads(2)
        assert torch.cuda.is_available()
        embedder = CLIPEmbedder()
        assert embedder.device.type == "cuda"
        assert next(embedder.model.parameters()).device.type == "cuda"
        fingerprint = embedder.fingerprint
        manager = DatabaseManager(str(Path(directory) / "test.db"))
        fixture_paths = []
        for frequency in (220, 440, 880):
            path = Path(directory) / f"tone-{frequency}.wav"
            samples = (np.sin(np.arange(48000) * (2 * np.pi * frequency / 48000)) * 16000).astype('<i2')
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(48000)
                audio.writeframes(samples.tobytes())
            source = get_audio_service().fingerprint(path)
            file_id = manager.upsert_file(AudioFileRecord(
                path=str(path), filename=path.name, duration=1.0,
                sample_rate=48000, channels=1, source_fingerprint=source.source_key,
            ))
            # Simulate the old text failure and a process interrupted mid-audio.
            manager.set_artifact_state("default", file_id, "text_vector", "failed", error_code="text_index_failed")
            fixture_paths.append(path)

        text = "BOOM Library long metadata description, impacts and explosions. " * 2000
        gpu_text = embedder.text_to_embedding(text)
        gpu_audio = [embedder.audio_to_embedding(str(p)) for p in fixture_paths]
        assert all(np.isfinite(v).all() for v in [gpu_text, *gpu_audio])
        embedder.model.to("cpu")
        embedder.device = torch.device("cpu")
        assert embedder._build_fingerprint() == fingerprint
        cpu_text = embedder.text_to_embedding(text)
        cpu_audio = [embedder.audio_to_embedding(str(p)) for p in fixture_paths]
        similarities = [float(np.dot(a, b)) for a, b in zip([gpu_text, *gpu_audio], [cpu_text, *cpu_audio])]
        assert min(similarities) > 0.999, similarities
        embedder.model.to("cuda")
        embedder.device = torch.device("cuda")

        # Exercise the real repair worker and real independent Chroma database.
        import main as backend
        from unittest import mock
        job_id = manager.create_job("default", "index_auto_gpu_test")
        with mock.patch.object(backend, "get_db_manager", return_value=manager):
            asyncio.run(backend._repair_index_task_locked(
                job_id, "default", ["waveform", "audio_vector", "text_vector"], "resume"
            ))
        job = manager.get_job(job_id)
        assert job["state"] == "completed", job
        assert (job["processed"], job["total"]) == (9, 9), job
        counts = manager.get_artifact_counts("default")
        assert all(counts[kind]["ready"] == 3 for kind in counts), counts
        report = {
            "python": sys.version,
            "test_data_dir": directory,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "model_device": str(embedder.device),
            "architecture": torch.cuda.get_device_capability(0),
            "fingerprint": fingerprint, "cpu_gpu_fingerprint_equal": True,
            "cpu_gpu_cosine_similarities": similarities,
            "long_text_characters": len(text), "max_text_tokens": embedder._text_max_length(),
            "job": job, "artifact_counts": counts,
        }
        output = scratch / "gpu-verification.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        from core.indexer import reset_all_indexers, reset_chroma_client
        reset_all_indexers()
        reset_chroma_client()
        backend._model_executor.shutdown(wait=True)
        backend._executor.shutdown(wait=True)
        manager._get_connection().close()
        reset_embedder()
        logging.shutdown()


if __name__ == "__main__":
    main()
