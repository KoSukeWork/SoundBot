# Data directory

SoundBot stores `db/`, `chroma_projects/`, and the metadata repair progress
database in its data directory. Configuration, encrypted secrets, logs, Electron
cache and temporary clips remain under the existing user profile or their own
configured locations. Audio source files and model weights are not moved.

The location is resolved in this order:

1. `SOUNDBOT_DATA_DIR` environment variable (absolute path).
2. `dataDir` in the profile's `user_config.json` (absolute path).
3. The existing userData directory, preserving previous default behavior.

Use **Settings → Change data directory**, next to the temporary directory setting.
Pause import/indexing in every project and wait for operations to finish. The app
checks for idle work, shuts its backend down gracefully, copies and validates the
data, saves the setting, verifies backend startup, and resumes unfinished work.
If the destination already contains a library, explicitly choose to use that
library or cancel. Existing data is never silently overwritten. An environment
override disables persistent changes; remove it and restart before changing the
setting.

## Offline CLI

Normally exit every SoundBot instance first. The CLI refuses migration while any
recognized SoundBot backend is running, including an independent test instance.
Do not launch another app instance while migration or rollback is running.

```text
python scripts/move_data_dir.py --to <absolute-new-directory> --dry-run
python scripts/move_data_dir.py --to <absolute-new-directory>
```

Use the same Python environment as the source backend. Alternatively, a packaged
backend exposes `soundbot-backend --storage-tool` with the same arguments. Add
`--user-data-dir <profile-directory>` when using a nondefault profile; this is the
profile containing the configuration, not the new data directory.

`--dry-run` checks process state, paths, estimated space and permissions without
creating the target or changing configuration. Actual migration also validates
the databases and copied bytes. Remove `SOUNDBOT_DATA_DIR` before persisting a
change, so it cannot override the new saved location. To explicitly use a library
already present at the destination, add `--use-existing`; no libraries are merged.

Migration requires source size plus the larger of 10% or 64 MiB of free space.
It copies only managed storage items to an owned staging directory, compares file
lists, sizes and SHA-256 hashes, runs SQLite `quick_check`, and compares all
user-table row counts (including Chroma SQLite and repair progress). Source and
destination must be separate, non-nested directories; mapped/symlink destinations
are rejected. No database schema or model changes are part of migration.

After verification, old items are renamed to `*.migrated-<timestamp>-<id>` and a
notice is written in the old location. Configuration is replaced atomically last.
Old backups are retained indefinitely; their eventual removal is a separate user
decision. Hashing large libraries requires several full reads and can take time.

## Recovery

Ordinary failures restore original names and configuration and remove only the
current transaction's incomplete copies. If a process is interrupted, the pending
`data-dir-migration.json` blocks backend startup. With all backends offline, run:

```text
python scripts/move_data_dir.py --recover
```

To undo the last completed migration, use the ID printed by the migration result
(also recorded in `data-dir-last-migration.json`):

```text
python scripts/move_data_dir.py --rollback <transaction-id>
```

Rollback restores the original library and configuration. It retains the copied
destination items as `*.rolled-back-<id>`; with `--use-existing`, the destination
library stays in place. Work completed in the new library after migration is not
merged back into the old library. Pause and exit normally first. If rollback itself
is interrupted, `--recover` continues it while preserving the copied data. Do not
manually remove the journal or rename conflicting directories to bypass checks.

If a desktop validation restart fails, the app attempts this rollback before
restarting the old library. It refuses to roll back while the new backend cannot
exit gracefully. If the resume response is lost after successful validation, the
new directory stays selected; a normal app restart resumes work.

**Do not run an old installed SoundBot after migration.** Old versions do not
understand `dataDir` or the migration notice and can create an empty library in
the old location. Use a source/package version containing this feature.

The GPU launcher accepts `-DataDir` for an explicit environment override; normally
omit it to honor the saved setting. Model location uses `-ModelsPath` or
`SOUNDBOT_MODELS_PATH`, then common repository, profile, installed and portable
locations. No machine-specific model path is required in the launcher.
