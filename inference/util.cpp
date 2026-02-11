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

#include "util.h"

using namespace std;

double total_num = 0;
double mpc_num = 0;
double mpc_fpga_num = 0;

json read_json(string filename) {
    ifstream f(filename);
    if (!f.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }
    json data;
    f >> data;
    return data;
}

uint64_t mod_sub(uint64_t x, uint64_t y, uint64_t mod, bool is_print) {
    double res = double(x) - double(y);
    if (is_print) {
        printf("res mod=%f\n", res);
        printf("res mod add=%lu\n", uint64_t(res + (double)mod));
    }
    uint64_t res_mod = uint64_t(res + (double)mod) % mod;
    if (is_print) {
        printf("res mod 97=%lu\n", res_mod);
    }
    return res_mod;
}

double uint64_to_double(uint64_t input, double scale, uint64_t ring_mod) {
    double output = 0;
    if (input < ring_mod / 2) {
        output = (double)input / scale;
    } else {
        output = -(double)(ring_mod - input) / scale;
    }
    return output;
}

uint64_t double_to_uint64(double input, double scale, uint64_t ring_mod) {
    uint64_t output = 0;
    input = input * scale;
    if (input >= 0) {
        output = (uint64_t)input;
    } else {
        output = ring_mod - ((uint64_t)(-input));
    }
    return output;
}

void vec_to_share(Array1D& vec, Array1DUint& share1, Array1DUint& share2, int scale_ord, uint64_t ring_mod) {
    double scale = pow(2, scale_ord);
    for (int j = 0; j < vec.size(); j++) {
        uint64_t temp0 = double_to_uint64(vec[j], scale, ring_mod);
        auto random = 0;
        share1[j] = (temp0 + random) % ring_mod;
        share2[j] = ring_mod - random;
    }
}

// Flip spatial dimensions + transpose input/output channels
Array<double, 4> transpose_weight(const Array<double, 4>& weight) {
    auto shape = weight.get_shape();
    uint32_t IC = shape[0];
    uint32_t OC = shape[1];
    uint32_t kH = shape[2];
    uint32_t kW = shape[3];

    Array<double, 4> transformed({OC, IC, kH, kW});

    for (int ic = 0; ic < IC; ic++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int flipped_kh = kH - 1 - kh;
                    int flipped_kw = kW - 1 - kw;

                    double temp_value = weight.get(ic, oc, flipped_kh, flipped_kw);
                    transformed.set(oc, ic, kh, kw, temp_value);
                }
            }
        }
    }
    return transformed;
}

void print_array(const Array<double, 3>& arr, std::ostream& out) {
    auto shape = arr.get_shape();
    int C = shape[0];
    int H = shape[1];
    int W = shape[2];

    out << "Array shape: [" << C << ", " << H << ", " << W << "]\n";
    for (int c = 0; c < C; c++) {
        out << "Channel " << c << ":\n";
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                out << arr.get(c, h, w) << " ";
            }
            out << "\n";
        }
        out << "\n";
    }
    out << std::flush;
}

void print_array_to_file(const Array<double, 3>& arr, const std::string& filename, bool append) {
    std::ios_base::openmode mode = std::ios::out;
    if (append) {
        mode |= std::ios::app;
    }

    std::ofstream outfile(filename, mode);

    if (!outfile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    auto shape = arr.get_shape();
    int C = shape[0];
    int H = shape[1];
    int W = shape[2];

    outfile << "Array shape: [" << C << ", " << H << ", " << W << "]\n";
    for (int c = 0; c < C; c++) {
        outfile << "Channel " << c << ":\n";
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                outfile << arr.get(c, h, w) << " ";
            }
            outfile << "\n";
        }
        outfile << "\n";
    }
    outfile.close();
}

Array<double, 3> upsample_with_zero(const Array<double, 3>& x, const Duo& stride) {
    auto x_shape = x.get_shape();
    uint32_t C = x_shape[0];
    int H = x_shape[1];
    int W = x_shape[2];

    uint32_t H_new = H * stride[0];
    uint32_t W_new = W * stride[1];

    Array<double, 3> out({C, H_new, W_new});

    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H_new; h++) {
            for (int w = 0; w < W_new; w++) {
                out.set(c, h, w, 0.0);
            }
        }
    }
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                out.set(c, h * stride[0], w * stride[1], x.get(c, h, w));
            }
        }
    }
    return out;
}
