# PolyBase Stockmeyer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename the generic Stockmeyer planner to `PolyBase` and add a tested Stockmeyer execution path to `PolyRelu2D` for ordinary and multiplexed packing.

**Architecture:** Keep Stockmeyer scale/level planning in a coefficient-independent `PolyBase` layer helper. `ParUpperDiagonalPoly` continues to own diagonal coefficient packing, while `PolyRelu2D` owns a `PolyBase` planner member and reuses its own packing builders. Python mirrors the planner and operation graph.

**Tech Stack:** C++17, CKKS context APIs, Catch2 heterogeneous tests, Python model-generator DAG nodes, pytest/unittest discovery.

---

### Task 1: Rename and decouple the planner

**Files:**
- Rename `inference/fhe_layers/mat_poly_base.h` → `inference/fhe_layers/poly_base.h`
- Rename `inference/fhe_layers/mat_poly_base.cpp` → `inference/fhe_layers/poly_base.cpp`
- Modify `inference/fhe_layers/CMakeLists.txt`, `fhe_layers.h`, `par_upper_diagonal_poly.h/.cpp`, `test_fhe_layers_hetero.cpp`
- Rename `inference/model_generator/layers/mat_poly_base.py` → `poly_base.py`
- Modify `inference/model_generator/layers/par_upper_diagonal_poly.py`, `training/model_compiler/transforms.py`

- [ ] Rename symbols and includes globally.
- [ ] Change `MatPolyBase` constructor to `(const CkksParameter&, uint32_t level, int order)` and remove `Array<double,2>` storage.
- [ ] Keep all existing normal and Horner planner methods, changing diagnostics to `PolyBase`.
- [ ] Update `ParUpperDiagonalPoly` to store coefficients itself and initialize `PolyBase` with parameter/level/order.
- [ ] Update Python class/module names and all compiler imports.
- [ ] Run `rg -n 'MatPolyBase|mat_poly_base'` and require no matches.
- [ ] Build the planner and existing ParUpperDiagonal tests.

### Task 2: Add failing C++ Stockmeyer PolyRelu2D test

**Files:**
- Modify `inference/unittests/test_fhe_layers_hetero.cpp`

- [ ] Copy the existing `poly_relu_bsgs_feature2d` test with the same degree-7 coefficient vectors and numerical checks.
- [ ] Use `compute_stockmeyer_level_cost`, `prepare_weight_stockmeyer`, Stockmeyer generated task path, and Stockmeyer output level.
- [ ] Run the focused test before implementation; confirm compilation/API failure.

### Task 3: Implement C++ PolyRelu2D planner and execution

**Files:**
- Modify `inference/fhe_layers/poly_relu2d.h/.cpp`

- [ ] Include `poly_base.h` and add a `PolyBase stockmeyer_base_` member initialized with layer parameter, level, and order.
- [ ] Add `prepare_weight_stockmeyer`, lazy variant, and `generate_weight_pt_for_stockmeyer`; share the existing ordinary/multiplexed slot layout with BSGS generation and use planner coefficient scale/level maps.
- [ ] Add `run_core_stockmeyer` using planner powers, four-term baby polynomials, planner target levels/scales, and giant-power combination; support lazy and eager plaintexts.
- [ ] Add `run_stockmeyer` preserving `Feature2DEncrypted` metadata and output level.
- [ ] Make `prepare_weight`/`prepare_weight_lazy` continue existing behavior unless explicitly called by tests.
- [ ] Run the focused C++ test and fix scale/level alignment issues until it passes.

### Task 4: Add Python Stockmeyer DAG support and test

**Files:**
- Modify `inference/model_generator/layers/poly_relu_base.py`, `poly_relu2d.py`
- Modify `inference/unittests/test_gen_layers.py`

- [ ] Add planner-backed Stockmeyer core with the same four-term grouping and combine levels as C++.
- [ ] Add eager `call_stockmeyer_feature2d` and lazy/custom-compute equivalent with `poly_relu_stockmeyer` op naming.
- [ ] Copy `test_poly_relu_bsgs_feature2d`, preserve coefficients and final sign-times-input graph, and switch cost/API/path names.
- [ ] Run the focused Python test and inspect generated argument names/levels.

### Task 5: Full validation and cleanup

**Files:** Existing files from Tasks 1–4.

- [ ] Build `fhe_layers` and heterogeneous unit-test targets.
- [ ] Run focused C++ Stockmeyer tests and existing ParUpperDiagonal Stockmeyer/Horner tests.
- [ ] Run focused Python generator tests.
- [ ] Run formatting/lint checks required by the repository.
- [ ] Re-run `rg` for stale names and review `git diff` for accidental Horner removal or unrelated changes.
- [ ] Commit implementation and tests with a focused message.
