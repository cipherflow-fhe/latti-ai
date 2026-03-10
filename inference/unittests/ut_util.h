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

#include <vector>
#include "common.h"

struct ArrayComparison {
    int dim;
    double max_abs;
    double max_error;
    std::vector<int> max_error_pos;
    double rms;
    double rmse;
};

ArrayComparison compare(const Array2D& expected, const Array2D& output);

ArrayComparison compare(const Array3D& expected, const Array3D& output);

ArrayComparison compare(const Array<double, 3>& expected, const Array<double, 3>& output);

ArrayComparison compare(const Array<double, 2>& expected, const Array<double, 2>& output);

ArrayComparison compare(const Array1D& expected, const Array1D& output);

ArrayComparison compare(const Array<double, 1>& expected, const Array<double, 1>& output);

void save_csv_3d(std::vector<std::vector<std::vector<double>>> x, std::string file_path);
