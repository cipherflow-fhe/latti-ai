// Common browser entry point for task-specific acceptance flows.
// Set LATTI_E2E_TASK=retina or resnet20; task scripts remain in their own
// examples/task directory so they can also be shipped independently.
const task = (process.env.LATTI_E2E_TASK || 'retina').toLowerCase();
if (task === 'retina') await import('../test_retina_dkd/run_browser.mjs');
else if (task === 'resnet20' || task === 'cifar10') await import('../../inference/interface/wasm/tests/workflow.mjs');
else throw new Error(`Unknown LATTI_E2E_TASK: ${task}`);
