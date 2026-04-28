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

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "interface/inference_server.h"

namespace py = pybind11;

// ── py::bytes ↔ Bytes (vector<uint8_t>) conversion ──────────────────────────

static py::bytes to_pybytes(const Bytes& b) {
    return py::bytes(reinterpret_cast<const char*>(b.data()), b.size());
}

static Bytes from_pybytes(const py::bytes& b) {
    const char* ptr = PyBytes_AS_STRING(b.ptr());
    Py_ssize_t len = PyBytes_GET_SIZE(b.ptr());
    return Bytes(reinterpret_cast<const uint8_t*>(ptr), reinterpret_cast<const uint8_t*>(ptr) + len);
}

static py::dict to_pybytes_dict(const std::map<std::string, Bytes>& m) {
    py::dict d;
    for (auto& [k, v] : m) {
        d[py::cast(k)] = to_pybytes(v);
    }
    return d;
}

static std::map<std::string, Bytes> from_pybytes_dict(const py::dict& d) {
    std::map<std::string, Bytes> m;
    for (auto& [key, val] : d) {
        m[key.cast<std::string>()] = from_pybytes(val.cast<py::bytes>());
    }
    return m;
}

PYBIND11_MODULE(latti_server, m) {
    m.doc() = "Python bindings for FHE inference server (import context, load model, evaluate)";

    py::class_<InferenceServer>(m, "InferenceServer")
        .def(py::init([](const std::string& server_dir, bool use_gpu, int gpu_device) {
                 py::gil_scoped_release release;
                 return InferenceServer(server_dir, use_gpu, gpu_device);
             }),
             py::arg("server_dir"), py::arg("use_gpu") = false, py::arg("gpu_device") = 0)
        .def(
            "import_eval_context",
            [](InferenceServer& self, const py::bytes& eval_ctx) {
                auto data = from_pybytes(eval_ctx);
                py::gil_scoped_release release;
                self.import_eval_context(data);
            },
            py::arg("eval_context"))
        .def("load_model", &InferenceServer::load_model, py::call_guard<py::gil_scoped_release>())
        .def(
            "evaluate",
            [](InferenceServer& self, const py::dict& encrypted_inputs, py::object progress_callback) {
                auto inputs = from_pybytes_dict(encrypted_inputs);
                lattisense::ProgressCallback cpp_cb = nullptr;
                if (!progress_callback.is_none()) {
                    // Prevent the Python callback from being garbage collected during inference.
                    auto cb_ref = std::make_shared<py::object>(progress_callback);
                    cpp_cb = [cb_ref](int completed, int total) {
                        py::gil_scoped_acquire acquire;
                        (*cb_ref)(completed, total);
                    };
                }
                std::map<std::string, Bytes> result;
                {
                    py::gil_scoped_release release;
                    result = self.evaluate(inputs, cpp_cb);
                }
                return to_pybytes_dict(result);
            },
            py::arg("encrypted_inputs"), py::arg("progress_callback") = py::none())
        .def("evaluate_plaintext", &InferenceServer::evaluate_plaintext, py::arg("input_csvs"),
             py::call_guard<py::gil_scoped_release>());
}
