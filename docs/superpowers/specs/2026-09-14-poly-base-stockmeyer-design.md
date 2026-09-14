# PolyBase Stockmeyer polynomial evaluation design

## Goal

Rename the generic polynomial scale/level planner from `MatPolyBase`/`mat_poly_base` to `PolyBase`/`poly_base`, remove its dependency on matrix coefficient storage, and reuse it for `PolyRelu2D` Stockmeyer evaluation. Existing `ParUpperDiagonalPoly` behavior remains unchanged, including its existing Stockmeyer Horner path. New `PolyRelu2D` APIs support ordinary and multiplexed packing and the non-Horner Stockmeyer algorithm only.

## Architecture

`PolyBase` remains a `Layer`-derived planning helper so it can share CKKS parameters and level state with existing C++ layers, but owns only polynomial order and Stockmeyer planning state. It no longer stores `Array<double, 2>` coefficients. Its public/protected API continues to provide level-cost calculation, initialization, power metadata, coefficient scale/level maps, baby-polynomial target metadata, and output level. Horner planning remains available for `ParUpperDiagonalPoly` compatibility.

`ParUpperDiagonalPoly` inherits `PolyBase` and keeps coefficient data and diagonal packing in its own class. Its includes, constructor, and build target use `poly_base.*`.

`PolyRelu2D` owns a `PolyBase` planner initialized with the same CKKS parameter, level, and order as the layer. Its Stockmeyer weight preparation uses the existing ordinary/multiplexed slot construction, with coefficient scales and target levels obtained from the planner. Its execution builds powers and four-term baby polynomials, then combines giant polynomials using the planner's target levels/scales. Lazy mode generates the same plaintext nodes on demand.

## Python model generator

Rename `mat_poly_base.py` and `MatPolyBase` to `poly_base.py` and `PolyBase`; update `ParUpperDiagonalPoly` and compiler transform imports. Add `PolyRelu2D.call_stockmeyer_feature2d()` and lazy generation using the same operation graph as C++ and the same generic planner metadata. Do not add Stockmeyer Horner APIs to `PolyRelu2D`.

## Tests

Add Python and heterogeneous C++ `poly_relu_stockmeyer_feature2d` tests mirroring the existing BSGS feature2d test, including the same degree-7 coefficients, level calculation, sign-times-input final operation, and numerical tolerances. Add planner/name checks through the existing ParUpperDiagonal Stockmeyer tests after renaming.

## Compatibility and validation

The old header, source filename, class name, and Python module/class name are removed. All repository references are updated. Build `fhe_layers` and the heterogeneous unit test target, run the focused C++ tests, and run the focused Python generator tests.
