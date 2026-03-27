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

#include <string>

#include <cxx_sdk_v2/cxx_fhe_task.h>

namespace ls = cxx_sdk_v2;

std::string fpga_base_path = "/acc_test/integrate/noc_config_16c_3";
std::string gpu_base_path = "/acc_test/integrate/gpu_tests";

class CkksFixture {
public:
    CkksFixture()
        : N{8192}, n_slot{N / 2}, param{ls::CkksParameter::create_parameter(8192)},
          context{ls::CkksContext::create_random_context(param)}, min_level{1}, max_level{param.get_max_level()},
          default_scale{param.get_default_scale()} {}

protected:
    int N;
    int n_slot;
    ls::CkksParameter param;
    ls::CkksContext context;
    int min_level;
    int max_level;
    double default_scale;
};

class CpuFixture {
public:
    CpuFixture() {}
    ~CpuFixture() {}
};

class BfvCpuFixture : public CpuFixture {
public:
    BfvCpuFixture()
        : n{8192 * 2}, t{65537}, param{ls::BfvParameter::create_parameter(n, t)},
          ctx{ls::BfvContext::create_random_context(param)}, n_op{4}, min_level{1}, max_level{param.get_max_level()} {}

protected:
    uint64_t n;
    uint64_t t;
    ls::BfvParameter param;
    ls::BfvContext ctx;
    int n_op;
    int min_level;
    int max_level;
};

class CkksCpuFixture : public CpuFixture {
public:
    CkksCpuFixture()
        : N{16384}, n_slot{N / 2}, param{ls::CkksParameter::create_parameter(N)},
          context{ls::CkksContext::create_random_context(param)}, min_level{1}, max_level{param.get_max_level()},
          default_scale{param.get_default_scale()} {
        context.gen_rotation_keys();
    }

protected:
    int N;
    int n_slot;
    ls::CkksParameter param;
    ls::CkksContext context;
    int min_level;
    int max_level;
    double default_scale;
};

class GpuFixture {
public:
    GpuFixture() {}

    ~GpuFixture() {}
};

class BfvGpuFixture : public GpuFixture {
public:
    BfvGpuFixture()
        : n{16384}, t{65537}, param{ls::BfvParameter::create_parameter(n, t)},
          ctx{ls::BfvContext::create_random_context(param)}, n_op{4}, min_level{1}, max_level{param.get_max_level()} {}

protected:
    uint64_t n;
    uint64_t t;
    ls::BfvParameter param;
    ls::BfvContext ctx;
    int n_op;
    int min_level;
    int max_level;
};

class CkksGpuFixture : public GpuFixture {
public:
    CkksGpuFixture()
        : N{16384}, n_slot{N / 2}, param{ls::CkksParameter::create_parameter(N)},
          context{ls::CkksContext::create_random_context(param)}, min_level{0}, max_level{param.get_max_level()},
          default_scale{param.get_default_scale()} {
        context.gen_rotation_keys();
    }

protected:
    int N;
    int n_slot;
    ls::CkksParameter param;
    ls::CkksContext context;
    int min_level;
    int max_level;
    double default_scale;
};

class LattigoCkksBtpFixture {
public:
    LattigoCkksBtpFixture()
        : param{ls::CkksBtpParameter::create_parameter()}, context{ls::CkksBtpContext::create_random_context(param)},
          default_scale{param.get_default_scale()} {
        n = param.get_n();
        n_slot = n / 2;
    }

protected:
    int n;
    int n_slot;
    ls::CkksBtpParameter param;
    ls::CkksBtpContext context;
    double default_scale;
};
