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

#include <vector>
#include <cmath>
#include <iostream>

#include "ut_util.h"

using namespace std;

ArrayComparison compare(const Array2D& expected, const Array2D& output) {
    double max_error = 0.0;
    double max_abs = 0.0;
    int max_error_pos[2] = {0, 0};
    double squared_error = 0.0;
    double squared = 0.0;
    vector<size_t> shape = {expected.size(), expected[0].size()};
    for (int i0 = 0; i0 < shape[0]; i0++) {
        for (int i1 = 0; i1 < shape[1]; i1++) {
            double y_pc = expected[i0][i1];
            double y = output[i0][i1];
            double diff = fabs(y_pc - y);
            squared_error += (y_pc - y) * (y_pc - y);
            squared += y_pc * y_pc;
            if (max_error < diff) {
                max_error = diff;
                max_error_pos[0] = i0;
                max_error_pos[1] = i1;
            }
            if (max_abs < fabs(y_pc)) {
                max_abs = fabs(y_pc);
            }
        }
    }

    ArrayComparison result;
    result.dim = 2;
    result.max_abs = max_abs;
    result.max_error = max_error;
    result.max_error_pos = {max_error_pos[0], max_error_pos[1]};
    result.rms = sqrt(squared / (shape[0] * shape[1]));
    result.rmse = sqrt(squared_error / (shape[0] * shape[1]));

    // Print max error position and values
    printf("Max error position: [%d, %d], expected=%.6f, actual=%.6f, error=%.6f\n", max_error_pos[0], max_error_pos[1],
           expected[max_error_pos[0]][max_error_pos[1]], output[max_error_pos[0]][max_error_pos[1]], max_error);

    return result;
}

ArrayComparison compare(const Array3D& expected, const Array3D& output) {
    double max_error = 0.0;
    double max_abs = 0.0;
    int max_error_pos[3] = {0, 0, 0};
    double squared_error = 0.0;
    double squared = 0.0;
    vector<size_t> shape = {expected.size(), expected[0].size(), expected[0][0].size()};
    for (int i0 = 0; i0 < shape[0]; i0++) {
        for (int i1 = 0; i1 < shape[1]; i1++) {
            for (int i2 = 0; i2 < shape[2]; i2++) {
                double y_pc = expected[i0][i1][i2];
                double y = output[i0][i1][i2];
                double diff = fabs(y_pc - y);
                squared_error += (y_pc - y) * (y_pc - y);
                squared += y_pc * y_pc;
                if (max_error < diff) {
                    max_error = diff;
                    max_error_pos[0] = i0;
                    max_error_pos[1] = i1;
                    max_error_pos[2] = i2;
                }
                if (max_abs < fabs(y_pc)) {
                    max_abs = fabs(y_pc);
                }
            }
        }
    }

    ArrayComparison result;
    result.dim = 3;
    result.max_abs = max_abs;
    result.max_error = max_error;
    result.max_error_pos = {max_error_pos[0], max_error_pos[1], max_error_pos[2]};
    result.rms = sqrt(squared / (shape[0] * shape[1] * shape[2]));
    result.rmse = sqrt(squared_error / (shape[0] * shape[1] * shape[2]));

    // Print max error position and values
    printf("Max error position: [%d, %d, %d], expected=%.6f, actual=%.6f, error=%.6f\n", max_error_pos[0],
           max_error_pos[1], max_error_pos[2], expected[max_error_pos[0]][max_error_pos[1]][max_error_pos[2]],
           output[max_error_pos[0]][max_error_pos[1]][max_error_pos[2]], max_error);

    return result;
}

ArrayComparison compare(const Array<double, 2>& expected, const Array<double, 2>& output) {
    double max_error = 0.0;
    double max_abs = 0.0;
    int max_error_pos[2] = {0, 0};
    double squared_error = 0.0;
    double squared = 0.0;
    array<uint64_t, 2> shape = expected.get_shape();
    for (int i0 = 0; i0 < shape[0]; i0++) {
        for (int i1 = 0; i1 < shape[1]; i1++) {
            double y_pc = expected.get(i0, i1);
            double y = output.get(i0, i1);
            double diff = fabs(y_pc - y);
            squared_error += (y_pc - y) * (y_pc - y);
            squared += y_pc * y_pc;
            if (max_error < diff) {
                max_error = diff;
                max_error_pos[0] = i0;
                max_error_pos[1] = i1;
            }
            if (max_abs < fabs(y_pc)) {
                max_abs = fabs(y_pc);
            }
        }
    }

    ArrayComparison result;
    result.dim = 2;
    result.max_abs = max_abs;
    result.max_error = max_error;
    result.max_error_pos = {max_error_pos[0], max_error_pos[1]};
    result.rms = sqrt(squared / (shape[0] * shape[1]));
    result.rmse = sqrt(squared_error / (shape[0] * shape[1]));

    // Print max error position and values
    printf("Max error position: [%d, %d], expected=%.6f, actual=%.6f, error=%.6f\n", max_error_pos[0], max_error_pos[1],
           expected.get(max_error_pos[0], max_error_pos[1]), output.get(max_error_pos[0], max_error_pos[1]), max_error);

    return result;
}

ArrayComparison compare(const Array<double, 3>& expected, const Array<double, 3>& output) {
    double max_error = 0.0;
    double max_abs = 0.0;
    int max_error_pos[3] = {0, 0, 0};
    double squared_error = 0.0;
    double squared = 0.0;
    array<uint64_t, 3> shape = expected.get_shape();
    for (int i0 = 0; i0 < shape[0]; i0++) {
        for (int i1 = 0; i1 < shape[1]; i1++) {
            for (int i2 = 0; i2 < shape[2]; i2++) {
                double y_pc = expected.get(i0, i1, i2);
                double y = output.get(i0, i1, i2);
                double diff = fabs(y_pc - y);
                squared_error += (y_pc - y) * (y_pc - y);
                squared += y_pc * y_pc;
                if (max_error < diff) {
                    max_error = diff;
                    max_error_pos[0] = i0;
                    max_error_pos[1] = i1;
                    max_error_pos[2] = i2;
                }
                if (max_abs < fabs(y_pc)) {
                    max_abs = fabs(y_pc);
                }
            }
        }
    }

    ArrayComparison result;
    result.dim = 3;
    result.max_abs = max_abs;
    result.max_error = max_error;
    result.max_error_pos = {max_error_pos[0], max_error_pos[1], max_error_pos[2]};
    result.rms = sqrt(squared / (shape[0] * shape[1] * shape[2]));
    result.rmse = sqrt(squared_error / (shape[0] * shape[1] * shape[2]));

    // Print max error position and values
    printf("Max error position: [%d, %d, %d], expected=%.6f, actual=%.6f, error=%.6f\n", max_error_pos[0],
           max_error_pos[1], max_error_pos[2], expected.get(max_error_pos[0], max_error_pos[1], max_error_pos[2]),
           output.get(max_error_pos[0], max_error_pos[1], max_error_pos[2]), max_error);

    return result;
}

ArrayComparison compare(const Array1D& expected, const Array1D& output) {
    double max_error = 0.0;
    double max_abs = 0.0;
    int max_error_pos[1] = {0};
    double squared_error = 0.0;
    double squared = 0.0;
    vector<size_t> shape = {expected.size()};
    for (int i0 = 0; i0 < shape[0]; i0++) {
        double y_pc = expected[i0];
        double y = output[i0];
        double diff = fabs(y_pc - y);
        squared_error += (y_pc - y) * (y_pc - y);
        squared += y_pc * y_pc;
        if (max_error < diff) {
            max_error = diff;
            max_error_pos[0] = i0;
        }
        if (max_abs < fabs(y_pc)) {
            max_abs = fabs(y_pc);
        }
    }

    ArrayComparison result;
    result.dim = 1;
    result.max_abs = max_abs;
    result.max_error = max_error;
    result.max_error_pos = {max_error_pos[0]};
    result.rms = sqrt(squared / shape[0]);
    result.rmse = sqrt(squared_error / shape[0]);

    // Print max error position and values
    printf("Max error position: [%d], expected=%.6f, actual=%.6f, error=%.6f\n", max_error_pos[0],
           expected[max_error_pos[0]], output[max_error_pos[0]], max_error);

    return result;
}

ArrayComparison compare(const Array<double, 1>& expected, const Array<double, 1>& output) {
    double max_error = 0.0;
    double max_abs = 0.0;
    int max_error_pos[1] = {0};
    double squared_error = 0.0;
    double squared = 0.0;
    auto shape = expected.get_shape();
    for (int i0 = 0; i0 < shape[0]; i0++) {
        double y_pc = expected.get(i0);
        double y = output.get(i0);
        double diff = fabs(y_pc - y);
        squared_error += (y_pc - y) * (y_pc - y);
        squared += y_pc * y_pc;
        if (max_error < diff) {
            max_error = diff;
            max_error_pos[0] = i0;
        }
        if (max_abs < fabs(y_pc)) {
            max_abs = fabs(y_pc);
        }
    }

    ArrayComparison result;
    result.dim = 1;
    result.max_abs = max_abs;
    result.max_error = max_error;
    result.max_error_pos = {max_error_pos[0]};
    result.rms = sqrt(squared / shape[0]);
    result.rmse = sqrt(squared_error / shape[0]);

    // Print max error position and values
    printf("Max error position: [%d], expected=%.6f, actual=%.6f, error=%.6f\n", max_error_pos[0],
           expected.get(max_error_pos[0]), output.get(max_error_pos[0]), max_error);

    return result;
}

void save_csv_3d(std::vector<std::vector<std::vector<double>>> x, std::string file_path) {
    std::ofstream outfile(file_path, std::ios::out);
    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < x[0].size(); j++) {
            for (int k = 0; k < x[0][0].size(); k++) {
                outfile << x[i][j][k] << ",";
            }
            outfile << std::endl;
        }
        outfile << std::endl;
    }
}
