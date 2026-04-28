# Plan: Real Progress Reporting + Dynamic GPU Device Selection

## Current State

lattisense SDK already supports both features at the lowest level:
- `FheTaskGpu::run(ctx, args, progress_cb, gpu_device)` — callback + device parameter
- `FheTaskCpu::run(ctx, args, progress_cb)` — callback parameter
- `ProgressCallback = std::function<void(int completed, int total)>` — defined in `cxx_fhe_task.h`

But the upper layers (`InferenceProcess` → `InferenceServer` → pybind11 → worker.py`) don't pass these parameters through.

## Changes (bottom-up)

### Step 1: `InferenceProcess` — pass callback and gpu_device through

**File:** `inference/inference_task/inference_process.h`
- Add `int gpu_device_ = 0;` member to `InferenceProcess`
- Change `run_task_lazy()` signature: `void run_task_lazy(bool is_mpc = false, lattisense::ProgressCallback progress_cb = nullptr);`
- Change `run_task()` signature similarly (for completeness, though server uses lazy)

**File:** `inference/inference_task/inference_process.cpp`
- In `run_task_lazy()`: pass `progress_cb` and `gpu_device_` to `fhe_task_gpu_->run()` / `fhe_task_cpu_->run()`
- In `run_task()`: same change

### Step 2: `InferenceServer` — expose callback and gpu_device

**File:** `inference/interface/inference_server.h`
- Add `int gpu_device_ = 0;` member
- Change constructor: `InferenceServer(const std::string& server_dir, bool use_gpu = false, int gpu_device = 0);`
- Change `evaluate()` signature: add `ProgressCallback progress_cb = nullptr`

**File:** `inference/interface/inference_server.cpp`
- Constructor: store `gpu_device_`
- `load_model()`: set `fp_->gpu_device_ = gpu_device_;`
- `evaluate()`: pass `progress_cb` to `fp_->run_task_lazy(false, progress_cb)`

### Step 3: pybind11 bindings — expose to Python

**File:** `inference/interface/python/server/bindings.cpp`
- Constructor: add `gpu_device` parameter
- `evaluate()`: accept optional `progress_callback` (Python callable)
  - Wrap Python callable in `ProgressCallback` lambda with GIL acquire
  - Important: must acquire GIL before calling Python function from C++ worker thread

### Step 4: worker.py — use real progress

**File:** `latti-client/server/worker.py`
- `ModelInstance.__init__()`: accept `gpu_device` parameter from config
- `ModelInstance.ensure_server()`: pass `gpu_device` to `InferenceServer()`
- `process_task()`: remove the estimated-time background thread
- `process_task()`: pass a real progress callback to `model.server.evaluate()`
- The callback: update `case.progress` and push SSE event via `store.push_event()`
- Note: callback runs from C++ thread with GIL acquired, so Redis/store calls must be thread-safe (they are — Redis client is inherently thread-safe)

### Step 5: config.toml — add gpu_device

**File:** `latti-client/server/config.toml`
- Add `gpu_device = 0` under `[server]`

## Build & Test

1. Rebuild latti-ai: `cd build && cmake --build . -j$(nproc)`
2. Rebuild pybind11: `cd inference/interface/python/build && cmake --build . -j$(nproc)`
3. Run E2E test: `cd /home/jiangsy/github/latti-client/server && python test_e2e.py`
4. Verify: real progress percentages in SSE events instead of time-based estimates

## File Change Summary

| File | Repo | Change |
|------|------|--------|
| `inference_process.h` | latti-ai | Add `gpu_device_`, update `run_task_lazy` signature |
| `inference_process.cpp` | latti-ai | Pass callback+device to FheTask::run() |
| `inference_server.h` | latti-ai | Add `gpu_device_`, update constructor + evaluate |
| `inference_server.cpp` | latti-ai | Pass through gpu_device and callback |
| `python/server/bindings.cpp` | latti-ai | Expose gpu_device + progress_callback to Python |
| `worker.py` | latti-client | Real progress callback, gpu_device from config |
| `config.toml` | latti-client | Add gpu_device setting |
