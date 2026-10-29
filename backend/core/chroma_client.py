"""Local Chroma 1.5.5 cache-sharding compatibility.

The Rust cache divides its capacity across 64 shards. On Windows the Python
default is 512 // 5 = 102, leaving ONE index per shard. A collision can evict
an unflushed index; eviction closes it without persisting or replaying its WAL.
Reserve the original collection budget in EACH shard, before bindings start.
This changes neither HNSW settings nor stored vectors. Revisit on Chroma upgrades.
"""
import chromadb
from chromadb.api import ServerAPI
from chromadb.api.client import Client
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import Settings, System


def persistent_client(path):
    settings = Settings(anonymized_telemetry=False, is_persistent=True,
                        persist_directory=str(path))
    if chromadb.__version__ != '1.5.5':
        return chromadb.PersistentClient(path=str(path), settings=settings)
    # Reuse an existing system rather than replace handles another client owns.
    existing = SharedSystemClient._identifier_to_system.get(str(path))
    if existing is not None:
        return Client(settings=settings)
    system = System(settings)
    api = system.instance(ServerAPI)
    # 102 collections per shard at the normal Windows descriptor limit.
    # Capacity reserves slots only: it does not load extra indexes or vectors.
    api.hnsw_cache_size *= 64
    try:
        system.start()
        return Client.from_system(system)
    except BaseException:
        system.stop()
        raise
