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
using Array1DUint = std::vector<uint64_t>;

using Bytes = std::vector<uint8_t>;

using Duo = std::array<uint32_t, 2>;

inline std::string str(const Duo& x) {
    return "(" + std::to_string(x[0]) + ',' + std::to_string(x[1]) + ')';
}
inline Duo operator+(const Duo& a, const Duo& b) {
    return {a[0] + b[0], a[1] + b[1]};
}
inline Duo operator-(const Duo& a, const Duo& b) {
    return {a[0] - b[0], a[1] - b[1]};
}
inline Duo operator*(const Duo& a, const Duo& b) {
    return {a[0] * b[0], a[1] * b[1]};
}
inline Duo operator*(const Duo& a, uint32_t s) {
    return {a[0] * s, a[1] * s};
}
inline Duo operator*(uint32_t s, const Duo& a) {
    return {s * a[0], s * a[1]};
}
inline Duo operator/(const Duo& a, const Duo& b) {
    return {a[0] / b[0], a[1] / b[1]};
}
inline Duo operator/(const Duo& a, uint32_t s) {
    return {a[0] / s, a[1] / s};
}
inline Duo operator%(const Duo& a, const Duo& b) {
    return {a[0] % b[0], a[1] % b[1]};
}
inline Duo operator%(const Duo& a, uint32_t s) {
    return {a[0] % s, a[1] % s};
}
inline uint32_t prod(const Duo& a) {
    return a[0] * a[1];
}
inline Duo div_mod(uint32_t p, uint32_t q) {
    return {p / q, p % q};
}
inline bool operator==(const Duo& a, const Duo& b) {
    return (a[0] == b[0]) && (a[1] == b[1]);
}
