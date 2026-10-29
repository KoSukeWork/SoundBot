"""Offline database migration; also used by the desktop's storage tool mode."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from core.data_migration import cli

if __name__ == '__main__':
    raise SystemExit(cli())
