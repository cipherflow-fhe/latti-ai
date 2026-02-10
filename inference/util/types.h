/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>
#include <array>

using Array1D = std::vector<double>;
using Array2D = std::vector<std::vector<double>>;
using Array3D = std::vector<std::vector<std::vector<double>>>;
using Array4D = std::vector<std::vector<std::vector<std::vector<double>>>>;
using Array1DUint = std::vector<uint64_t>;
using Array2DUint = std::vector<std::vector<uint64_t>>;
using Array3DUint = std::vector<std::vector<std::vector<uint64_t>>>;
using Array4DUint = std::vector<std::vector<std::vector<std::vector<uint64_t>>>>;

using Bytes = std::vector<uint8_t>;

using Duo = std::array<uint32_t, 2>;
