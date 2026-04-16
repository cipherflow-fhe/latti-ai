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
#include <memory>
#include <array>

template <typename T> using UPtr = std::unique_ptr<T>;

template <typename T, typename... Args> auto MakeU(Args&&... args) -> decltype(auto) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

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

struct DuoIterator {
    Duo current;
    Duo limit;

    const Duo& operator*() const {
        return current;
    }

    DuoIterator& operator++() {
        ++current[1];
        if (current[1] >= limit[1]) {
            current[1] = 0;
            ++current[0];
        }
        return *this;
    }

    bool operator==(const DuoIterator& other) const {
        return current == other.current && limit == other.limit;
    }

    bool operator!=(const DuoIterator& other) const {
        return !(*this == other);
    }
};

struct DuoRange {
    Duo limit;

    DuoIterator begin() const {
        if (limit[0] == 0 || limit[1] == 0) {
            return end();
        }
        return {{0, 0}, limit};
    }

    DuoIterator end() const {
        return {{limit[0], 0}, limit};
    }
};

inline DuoRange duo_range(const Duo& limit) {
    return {limit};
}
