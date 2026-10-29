"""Bounded CPU preparation and feature transport for continuous indexing."""
import multiprocessing
import queue
import time
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from core.audio_service import AudioService
from core.index_audio import prepare_index_audio, check_cancel, IndexWorkCancelled
from core.index_features import FeatureSpec, window_features
import numpy as np

_cancel = _features = _spec = None
_spool_limit = 0


def _timed_features(samples):
    start = time.perf_counter()
    value = window_features(samples, _spec)
    return value, time.perf_counter() - start


def _mel_windows(windows):
    count = max(1, min(2, int(os.getenv('SOUNDBOT_FEATURE_THREADS', '2'))))
    if count == 1:
        for samples in windows:
            check_cancel(_cancel)
            yield _timed_features(samples)
        return
    with ThreadPoolExecutor(max_workers=count) as executor:
        pending = deque()
        for _ in range(count):
            samples = next(windows, None)
            if samples is not None:
                pending.append(executor.submit(_timed_features, samples))
        while pending:
            check_cancel(_cancel)
            result = pending.popleft().result()
            samples = next(windows, None)
            if samples is not None:
                pending.append(executor.submit(_timed_features, samples))
            yield result


def _initialize(cancel, features, spec, spool_limit):
    global _cancel, _features, _spec, _spool_limit
    maximum, size, hop, window, filters, shape = spec
    spec = FeatureSpec(maximum, size, hop, np.frombuffer(window, dtype=np.float64),
                       np.frombuffer(filters, dtype=np.float64).reshape(shape))
    _cancel, _features, _spec, _spool_limit = cancel, features, spec, spool_limit


def _prepare(file_id, path, work_dir, rate, audio, waveform, need_metadata):
    from core.audio_metadata import probe_metadata
    service = AudioService(waveform_work_dir=work_dir)
    result = dict(metadata=None, waveform=None, windows=0, decode_error=None,
                  feature_error=None, cancelled=False, decode_seconds=0., mel_seconds=0.)
    prepared = windows = features_iter = None
    def metadata_ready(value):
        result['metadata'] = value
    try:
        check_cancel(_cancel)
        start = time.perf_counter()
        if audio or waveform:
            prepared = prepare_index_audio(service, path, rate, waveform=waveform,
                cancel=_cancel, need_metadata=need_metadata, max_spool_bytes=_spool_limit,
                on_metadata=metadata_ready)
            result['waveform'] = prepared.waveform
        else:
            if need_metadata:
                result['metadata'] = probe_metadata(service, path)
        result['decode_seconds'] = time.perf_counter() - start
    except IndexWorkCancelled:
        result['cancelled'] = True
    except Exception as exc:
        result['decode_error'] = exc
    try:
        if prepared is not None and audio:
            windows = prepared.windows(_spec.max_samples, _cancel)
            features_iter = _mel_windows(windows)
            for features, elapsed in features_iter:
                check_cancel(_cancel)
                result['mel_seconds'] += elapsed
                while True:
                    check_cancel(_cancel)
                    try:
                        _features.put((file_id, result['windows'], features), timeout=.05)
                        break
                    except queue.Full:
                        continue
                result['windows'] += 1
    except IndexWorkCancelled:
        result['cancelled'] = True
    except Exception as exc:
        result['feature_error'] = exc
    finally:
        if features_iter is not None:
            features_iter.close()
        if windows is not None:
            windows.close()
        if prepared is not None:
            prepared.close()
    return result


class FeaturePool:
    """At most workers source reads and 32 feature windows in transport.

    Only the initializer receives arrays describing the feature extractor, not
    the model or tokenizer. A job's workers are drained before claims restore.
    """
    def __init__(self, workers, spec, *, spool_limit=2 * 1024**3):
        context = multiprocessing.get_context('spawn')
        self.cancel = context.Event()
        self.features = context.Queue(maxsize=32)
        self.stats = {'decode_seconds': 0., 'mel_seconds': 0., 'windows': 0}
        # Large NumPy initializer arguments make Windows spawn wait for each
        # child to import NumPy before its pipe can accept the next pickle.
        # Shared read-only arrays avoid serializing those four cold starts.
        shared_spec = (spec.max_samples, spec.size, spec.hop,
                       context.RawArray('d', spec.window.reshape(-1)),
                       context.RawArray('d', spec.filters.reshape(-1)), spec.filters.shape)
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=_initialize, initargs=(self.cancel, self.features, shared_spec, spool_limit))

    def submit(self, file_id, path, service, rate, *, audio, waveform, need_metadata):
        check_cancel(self.cancel)
        return self.pool.submit(_prepare, file_id, path, str(service.waveform_work_dir),
                                rate, audio, waveform, need_metadata)

    def read_batch(self, size):
        """Collect only already-ready features with a short coalescing window."""
        result = []
        deadline = time.monotonic() + .05
        while len(result) < size and not self.cancel.is_set():
            try:
                result.append(self.features.get(timeout=max(.0001, deadline - time.monotonic())))
                if len(result) == 1:
                    deadline = time.monotonic() + .04
            except queue.Empty:
                break
            if time.monotonic() >= deadline:
                break
        return result

    def close(self):
        self.cancel.set()
        # Keep draining while joining: a worker's Queue feeder can otherwise
        # wait forever on a full pipe after the GPU consumer has stopped.
        import threading
        done = threading.Event()
        def drain():
            while not done.is_set():
                try:
                    self.features.get(timeout=.05)
                except queue.Empty:
                    pass
        thread = threading.Thread(target=drain, name='index-feature-drain')
        thread.start()
        try:
            self.pool.shutdown(wait=True, cancel_futures=True)
        finally:
            done.set()
            thread.join()
            self.features.close()
            self.features.join_thread()
