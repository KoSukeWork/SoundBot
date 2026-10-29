const test = require('node:test');
const assert = require('node:assert/strict');
const { switchStorage } = require('../../assets/main/storage-switch.cjs');

function fixture() {
  const calls = [];
  return { calls, services: {
    stop: async () => calls.push('stop'),
    tool: async args => { calls.push(args[0]); return { id: 'transaction', phase: 'complete' }; },
    start: async paused => { calls.push(paused ? 'start-paused' : 'start-old'); return { success: true }; },
    resume: async () => calls.push('resume')
  } };
}
test('verified restart precedes indexing resume', async () => {
  const { calls, services } = fixture();
  await switchStorage({ target: '/new' }, services);
  assert.deepEqual(calls, ['stop', '--to', 'start-paused', 'resume']);
});
test('busy backend prevents migration', async () => {
  const { calls, services } = fixture();
  services.stop = async () => { throw Error('pause first'); };
  await assert.rejects(switchStorage({ target: '/new' }, services), /pause first/);
  assert.deepEqual(calls, []);
});
test('copy failure restarts original library', async () => {
  const { calls, services } = fixture();
  services.tool = async () => { calls.push('copy-failed'); throw Error('disk full'); };
  await assert.rejects(switchStorage({ target: '/new' }, services), /disk full/);
  assert.deepEqual(calls, ['stop', 'copy-failed', 'start-old']);
});
test('validation startup failure shuts down then rolls back before restarting', async () => {
  const { calls, services } = fixture();
  services.start = async paused => { calls.push(paused ? 'bad-start' : 'start-old'); return { success: !paused, error: 'bad database' }; };
  await assert.rejects(switchStorage({ target: '/new' }, services), /bad database/);
  assert.deepEqual(calls, ['stop', '--to', 'bad-start', 'stop', '--rollback', 'start-old']);
});
test('failed graceful exit never proceeds to rollback', async () => {
  const { calls, services } = fixture(); let stopped = false;
  services.stop = async () => { if (stopped) throw Error('still running'); stopped = true; calls.push('stop'); };
  services.start = async () => ({ success: false });
  await assert.rejects(switchStorage({ target: '/new' }, services), /still running/);
  assert.deepEqual(calls, ['stop', '--to']);
});
