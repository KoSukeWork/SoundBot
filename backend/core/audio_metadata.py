"""Original-stream metadata shared by recovery and the offline repair tool."""
import math


def metadata_missing(record):
    return any(not isinstance(record.get(key), (int, float))
               or not math.isfinite(record[key]) or record[key] <= 0
               for key in ('duration', 'sample_rate', 'channels'))


def probe_metadata(service, path):
    from core.audio_service import AudioMetadata, AudioServiceError

    metadata = service.probe(path)
    # Some containers omit duration in their header. Decode only that case,
    # keeping the source sample rate/channel count (never CLAP's mono rate).
    if metadata_missing(metadata.to_dict()):
        decoded = service.decode(path)
        metadata = AudioMetadata(decoded.duration, decoded.sample_rate,
                                 decoded.channels, metadata.format, metadata.codec)
    if metadata_missing(metadata.to_dict()):
        raise AudioServiceError('audio_metadata_invalid', 'Audio has no valid stream properties')
    return metadata
