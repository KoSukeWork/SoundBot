"""Storage location and lifetime locks, independent of models and databases."""
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys


def resolve_data_dir(user_data, environ=None):
    env = os.environ if environ is None else environ
    user_data = Path(user_data).expanduser().resolve()
    value = env.get('SOUNDBOT_DATA_DIR')
    if not value:
        settings = user_data / 'user_config.json'
        if settings.exists():
            payload = json.loads(settings.read_text(encoding='utf-8-sig'))
            value = payload.get('dataDir')
    if not value:
        return user_data
    if not isinstance(value, str) or not Path(value).expanduser().is_absolute():
        raise ValueError('dataDir / SOUNDBOT_DATA_DIR must be an absolute path')
    return Path(value).expanduser().resolve()


@contextmanager
def directory_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.soundbot-storage.lock').open('a+b')
    locked = False
    try:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    except OSError as exc:
        if not locked:
            raise RuntimeError(f'SoundBot storage is in use: {root}') from exc
        raise
    finally:
        if locked:
            lock.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


@contextmanager
def storage_locks(*roots):
    with ExitStack() as stack:
        for root in sorted({Path(root).resolve() for root in roots}, key=str):
            stack.enter_context(directory_lock(root))
        yield


def require_backend_offline():
    """Fail closed before reading storage. The GUI may remain open without a backend."""
    if os.name == 'nt':
        result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command',
            '[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); '
            'Get-CimInstance Win32_Process | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress'],
            capture_output=True, text=True, encoding='utf-8', check=True,
            creationflags=subprocess.CREATE_NO_WINDOW)
        rows = json.loads(result.stdout)
        rows = rows if isinstance(rows, list) else [rows]
        commands = [(row['ProcessId'], row.get('Name') or '', row.get('CommandLine') or '') for row in rows]
    else:
        result = subprocess.run(['ps', '-eo', 'pid=,comm=,args='], capture_output=True, text=True, check=True)
        commands = []
        for line in result.stdout.splitlines():
            pid, name, command = line.strip().split(None, 2)
            commands.append((int(pid), name, command))
    for pid, name, command in commands:
        name, command = name.lower(), command.lower().replace('\\', '/')
        if pid == os.getpid() or '--storage-tool' in command:
            continue
        if 'soundbot-backend' in name or ('python' in name and
                ('backend/main.py' in command or ('soundbot' in command and 'main.py' in command))):
            raise RuntimeError(f'SoundBot backend is still running (PID {pid}); close it normally first')
