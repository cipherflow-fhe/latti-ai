// Common server-integration entry point. The selected task owns its model
// details; this wrapper only makes the two acceptance locations consistent.
const task = (process.env.LATTI_E2E_TASK || 'retina').toLowerCase();
if (task === 'retina') await import('../test_retina_dkd/run_server_integration.mjs');
else if (task === 'resnet20' || task === 'cifar10') await import('../../inference/interface/wasm/tests/workflow.mjs');
else throw new Error(`Unknown LATTI_E2E_TASK: ${task}`);
