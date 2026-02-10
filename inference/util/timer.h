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

#include <chrono>
#include <string>

class Timer {
public:
    Timer(bool s = false) : _duration{0} {
        if (s) {
            start();
        }
    }

    void start() {
        _start = std::chrono::high_resolution_clock::now();
    }

    Timer& stop() {
        auto end = std::chrono::high_resolution_clock::now();
        _duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - _start);
        return *this;
    }

    std::chrono::milliseconds& get_duration() {
        return _duration;
    }

    void print(const std::string& text = "") {
        fprintf(stderr, ">>> %s: %ld ms\n", text.c_str(), _duration.count());
    }

private:
    std::chrono::milliseconds _duration;
    std::chrono::_V2::system_clock::time_point _start;
};
