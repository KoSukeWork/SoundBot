"""Bounded, cancellable decode shared by waveform and CLAP indexing."""
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
import multiprocessing
import logging
import re
import shutil
import stat
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from core.audio_service import AudioService, WaveformResult, _av
from config import WAVEFORM_PEAK_COUNT

_worker_cancel = None
logger = logging.getLogger(__name__)
_SPOOL_NAME = re.compile(r'index-audio-[a-z0-9_]{8}\.f32\Z')
SPOOL_MIN_FREE_BYTES = 2 * 1024**3
SPOOL_SPACE_CHECK_BYTES = 64 * 1024**2


def default_parallelism():
    return max(1, min(4, os.cpu_count() or 1))


def _remove_spool(path):
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        # A traceback, decoder or antivirus may still hold a Windows handle.
        # Never replace the original cancellation/model/decode exception.
        logger.warning('Deferred index audio spool cleanup for %s: %s', path, exc)


def cleanup_stale_spools(root, *, now=None):
    """Remove only regular, owned spools older than 7 days, at startup.

    No recursion, symlinks, junctions, user clips, or arbitrary suffix matches.
    A week also avoids reclaiming normal in-flight work in another process.
    """
    root = Path(root)
    if (not root.exists() or root.is_symlink()
            or getattr(root, 'is_junction', lambda: False)()):
        return
    cutoff = (time.time() if now is None else now) - 7 * 86400
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        logger.warning('Unable to inspect index audio spool directory %s: %s', root, exc)
        return
    for path in entries:
        if not _SPOOL_NAME.fullmatch(path.name):
            continue
        try:
            info = path.lstat()
            if (stat.S_ISREG(info.st_mode) and not path.is_symlink()
                    and not getattr(info, 'st_file_attributes', 0) & 0x400
                    and info.st_mtime < cutoff):
                _remove_spool(path)
        except OSError as exc:
            logger.warning('Unable to inspect index audio spool %s: %s', path, exc)


def _init_decode_worker(cancel):
    global _worker_cancel
    _worker_cancel = cancel


def _decode_worker(work_dir, path, rate, waveform):
    service = AudioService(waveform_work_dir=work_dir)
    return prepare_index_audio(service, path, rate, waveform=waveform, cancel=_worker_cancel)


class DecodePool:
    """Bounded CPU decode pool; children do not load a model or run inference.

    Source entry points may still import Torch while Windows spawn imports main.
    """
    def __init__(self, workers):
        context = multiprocessing.get_context('spawn')
        self.cancel = context.Event()
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=_init_decode_worker, initargs=(self.cancel,))

    def prepare(self, service, path, rate, *, waveform, cancel=None):
        check_cancel(self.cancel)
        return self.pool.submit(_decode_worker, str(service.waveform_work_dir),
                                path, rate, waveform).result()

    def close(self):
        self.cancel.set()
        self.pool.shutdown(wait=True, cancel_futures=True)


class IndexWorkCancelled(Exception):
    """Cooperative cancellation, never an artifact failure."""


def check_cancel(event):
    if event is not None and event.is_set():
        raise IndexWorkCancelled()


def file_peaks(stream, size, cancel=None):
    """Exact old bin/max/normalization, with at most 1 MiB resident input."""
    if not size:
        return [0.] * WAVEFORM_PEAK_COUNT
    boundaries = np.linspace(0, size, WAVEFORM_PEAK_COUNT + 1, dtype=np.int64)
    peaks = np.zeros(WAVEFORM_PEAK_COUNT, dtype=np.float32)
    starts = np.minimum(boundaries[:-1], size - 1)
    ends = np.maximum(boundaries[1:], starts + 1)
    stream.seek(0)
    offset = 0
    while offset < size:
        check_cancel(cancel)
        count = min(size - offset, 262144)
        values = np.frombuffer(stream.read(count * 4), dtype=np.float32)
        if values.size != count:
            raise OSError('Truncated waveform spool')
        first = int(np.searchsorted(ends, offset, side='right'))
        last = int(np.searchsorted(starts, offset + count, side='left'))
        indices = np.maximum(starts[first:last] - offset, 0)
        # reduceat handles duplicate starts (tiny inputs with empty bins) by
        # taking that sample, exactly matching the original max(1, end-start).
        maxima = np.maximum.reduceat(np.abs(values), indices)
        np.maximum(peaks[first:last], maxima, out=peaks[first:last])
        offset += count
    maximum = float(np.max(peaks))
    if maximum > 0:
        peaks /= maximum
    return np.clip(peaks, 0., 1.).astype(float).tolist()


@dataclass
class IndexAudio:
    path: Path
    sample_rate: int
    waveform: object = None
    metadata: object = None

    def windows(self, size, cancel=None):
        # Read at most one window, even for multi-GB decoded streams. Do not
        # mmap the full CLAP stream: touched pages can inflate process RSS.
        with self.path.open('rb') as stream:
            while True:
                check_cancel(cancel)
                values = np.fromfile(stream, dtype=np.float32, count=size)
                if not values.size:
                    break
                yield values

    def close(self):
        _remove_spool(self.path)


def prepare_index_audio(service, path, sample_rate, *, waveform=True, cancel=None,
                        need_metadata=False, max_spool_bytes=None, on_metadata=None):
    source = service._validate_source(path)
    service._require_pyav()
    root = service.waveform_work_dir
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    spool_bytes = 0
    metadata = None
    try:
        # Both resamplers see the original decoded frames, preserving the
        # exact old mono/downsample paths, sample order and sanitisation.
        with tempfile.TemporaryFile(dir=root) as wave_sink:
            fd, name = tempfile.mkstemp(prefix='index-audio-', suffix='.f32', dir=root)
            raw = Path(name)
            paths.append(raw)
            count = 0
            with os.fdopen(fd, 'wb') as sink, _av.open(str(source), mode='r', metadata_errors='replace') as container:
                if need_metadata:
                    from core.audio_metadata import probe_metadata
                    metadata = probe_metadata(service, str(source), container=container)
                    if on_metadata is not None:
                        on_metadata(metadata)
                stream = next((s for s in container.streams if s.type == 'audio'), None)
                if stream is None:
                    raise ValueError('container has no audio stream')
                native_rate = int(stream.codec_context.sample_rate or stream.rate or 0)
                if native_rate < 1:
                    raise ValueError('audio stream has no sample rate')
                wave_resampler = _av.AudioResampler(format='fltp', layout='mono', rate=native_rate) if waveform else None
                clap_resampler = _av.AudioResampler(format='fltp', layout='mono', rate=sample_rate)
                next_space_check = 0

                def check_spool_space():
                    # Long recordings are bounded by the disk, not a fixed
                    # per-file budget; keep a reserve so the volume never fills.
                    nonlocal next_space_check
                    if spool_bytes < next_space_check:
                        return
                    next_space_check = spool_bytes + SPOOL_SPACE_CHECK_BYTES
                    if shutil.disk_usage(root).free < SPOOL_MIN_FREE_BYTES:
                        from core.audio_service import AudioServiceError
                        raise AudioServiceError('index_spool_disk_full',
                                                f'Not enough free disk space for the indexing spool in {root}')

                def write(resampler, frame, output):
                    nonlocal spool_bytes
                    size = 0
                    for converted in AudioService._iter_av_frames(resampler.resample(frame)):
                        values = np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
                        values = np.nan_to_num(values, copy=False, nan=0., posinf=1., neginf=-1.)
                        # Buffered writes avoid a descriptor/seek/flush for each
                        # tiny decoder frame (ndarray.tofile bypasses buffering).
                        spool_bytes += values.nbytes
                        if max_spool_bytes is not None and spool_bytes > max_spool_bytes:
                            from core.audio_service import AudioServiceError
                            raise AudioServiceError('index_spool_limit', 'Decoded audio exceeds the indexing spool budget')
                        check_spool_space()
                        output.write(values.tobytes())
                        size += values.size
                    return size

                for frame in AudioService._decode_audio_frames(container, stream):
                    check_cancel(cancel)
                    if wave_resampler is not None:
                        count += write(wave_resampler, frame, wave_sink)
                    write(clap_resampler, frame, sink)
                check_cancel(cancel)
                if wave_resampler is not None:
                    count += write(wave_resampler, None, wave_sink)
                write(clap_resampler, None, sink)
            result = None
            if waveform:
                wave_sink.flush()
                peaks = file_peaks(wave_sink, count, cancel)
                result = WaveformResult(peaks, count/native_rate, native_rate, service.fingerprint(source))
            check_cancel(cancel)
            return IndexAudio(raw, sample_rate, result, metadata)
    except BaseException:
        for item in paths:
            _remove_spool(item)
        raise
