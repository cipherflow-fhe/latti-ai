# Common browser/server acceptance entry points

The task-specific scripts remain alongside their task resources, while these
two wrappers provide one location for local acceptance commands:

```bash
LATTI_E2E_TASK=retina node examples/e2e/run_browser.mjs
LATTI_E2E_TASK=retina node examples/e2e/run_server_integration.mjs
LATTI_E2E_TASK=resnet20 node examples/e2e/run_browser.mjs
```

The ResNet20 workflow still uses the environment variables documented in
`inference/interface/wasm/tests/workflow.mjs`. The wrappers do not transform
ciphertext bytes or create ZIP archives; the 0.1.2 WASM server mode owns that
file-contract work.
