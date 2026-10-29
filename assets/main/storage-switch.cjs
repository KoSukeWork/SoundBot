// Desktop orchestration; filesystem migration lives exclusively in Python.
async function switchStorage(request, services) {
  let stopped = false;
  let migrated = null;
  try {
    await services.stop(); // Idle check + graceful exit. Never force-kill.
    stopped = true;
    migrated = await services.tool(['--to', request.target, ...(request.useExisting ? ['--use-existing'] : [])]);
    const started = await services.start(true); // Verify before automatic indexing.
    if (!started.success) throw new Error(started.error || '新数据目录的后端启动失败');
  } catch (error) {
    if (migrated) {
      // A failed validation backend must exit before restoring the old paths.
      await services.stop();
      await services.tool(['--rollback', migrated.id]);
    }
    if (stopped) {
      const restored = await services.start(false);
      if (!restored.success) throw new Error(`${error.message}；旧目录已恢复，但后端启动失败：${restored.error}`);
    }
    throw error;
  }
  try {
    await services.resume();
  } catch (error) {
    // A lost response may mean indexing already resumed. Do not roll back a
    // live, verified library; retain it and let a normal restart resume work.
    return { ...migrated, warning: `迁移已完成，续跑启动未确认：${error.message}。请正常重启应用。` };
  }
  return migrated;
}
module.exports = { switchStorage };
