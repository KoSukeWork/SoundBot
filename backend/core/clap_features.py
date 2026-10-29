"""Vectorized CPU form of CLAP's deterministic rand_trunc/repeatpad path.

Keeps HF's float64 windowing, complex64 FFT rounding, Slaney filters and dB
floor. Only the per-frame Python loop is removed; no GPU or mixed precision.
"""
import numpy as np
import torch


def window_inputs(audio, extractor, max_samples):
    from transformers.audio_utils import window_function, power_to_db
    audio = np.asarray(audio, dtype=np.float64)
    if not 0 < len(audio) <= max_samples:
        raise ValueError('Expected a nonempty, bounded CLAP window')
    if len(audio) < max_samples:
        audio = np.tile(audio, max_samples // len(audio))
        audio = np.pad(audio, (0, max_samples - len(audio)))
    size, hop = extractor.fft_window_size, extractor.hop_length
    padded = np.pad(audio, (size // 2, size // 2), mode='reflect')
    frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop]
    frames = frames * window_function(size, 'hann').astype(np.float64)
    spectrum = np.fft.rfft(frames, axis=-1).astype(np.complex64)
    power = (np.abs(spectrum, dtype=np.float64) ** 2.0).T.copy()
    # Triangular filters are sparse: each frequency contributes to at most two
    # mel bands. Avoid a dense BLAS product (and its nested thread pool).
    filters = extractor.mel_filters_slaney
    mel = np.zeros((filters.shape[1], power.shape[1]), dtype=np.float64)
    for band in range(filters.shape[1]):
        indices = np.flatnonzero(filters[:, band])
        if indices.size:
            first, last = indices[0], indices[-1] + 1
            mel[band] = np.einsum('f,ft->t', filters[first:last, band], power[first:last], optimize=False)
    mel = np.maximum(1e-10, mel)
    mel = np.asarray(power_to_db(mel), dtype=np.float32).T
    return {'input_features': torch.from_numpy(mel[None, None]),
            'is_longer': torch.tensor([[False]])}
