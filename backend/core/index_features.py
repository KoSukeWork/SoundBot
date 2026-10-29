"""NumPy-only CLAP preparation for spawned workers; no model/Torch imports."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FeatureSpec:
    max_samples: int
    size: int
    hop: int
    window: np.ndarray
    filters: np.ndarray

    @classmethod
    def from_embedder(cls, embedder):
        from transformers.audio_utils import window_function
        extractor = embedder.processor.feature_extractor
        return cls(embedder.max_samples, extractor.fft_window_size,
                   extractor.hop_length,
                   window_function(extractor.fft_window_size, 'hann').astype(np.float64),
                   extractor.mel_filters_slaney)


def window_features(audio, spec):
    # Same operations and rounding as clap_features.window_inputs, including
    # HF power_to_db's default reference=1, min_value=1e-10, db_range=None.
    audio = np.asarray(audio, dtype=np.float64)
    if not 0 < len(audio) <= spec.max_samples:
        raise ValueError('Expected a nonempty, bounded CLAP window')
    if len(audio) < spec.max_samples:
        audio = np.tile(audio, spec.max_samples // len(audio))
        audio = np.pad(audio, (0, spec.max_samples - len(audio)))
    padded = np.pad(audio, (spec.size // 2, spec.size // 2), mode='reflect')
    frames = np.lib.stride_tricks.sliding_window_view(padded, spec.size)[::spec.hop]
    spectrum = np.fft.rfft(frames * spec.window, axis=-1).astype(np.complex64)
    power = (np.abs(spectrum, dtype=np.float64) ** 2.0).T.copy()
    mel = np.zeros((spec.filters.shape[1], power.shape[1]), dtype=np.float64)
    for band in range(spec.filters.shape[1]):
        indices = np.flatnonzero(spec.filters[:, band])
        if indices.size:
            first, last = indices[0], indices[-1] + 1
            mel[band] = np.einsum('f,ft->t', spec.filters[first:last, band],
                                  power[first:last], optimize=False)
    mel = np.maximum(1e-10, mel)
    mel = 10.0 * (np.log10(np.clip(mel, 1e-10, None)) - np.log10(1.0))
    return np.asarray(mel, dtype=np.float32).T[None, None]
