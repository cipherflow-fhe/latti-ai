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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "interface/inference_client.h"

namespace py = pybind11;

// ── py::bytes ↔ Bytes (vector<uint8_t>) conversion ──────────────────────────
// pybind11's stl.h type caster does NOT auto-convert Python bytes ↔ vector<uint8_t>.
// We must handle this explicitly.

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

PYBIND11_MODULE(latti_client, m) {
    m.doc() = "Python bindings for FHE inference client (key gen, encrypt, decrypt)";

    py::class_<DecryptedOutput>(m, "DecryptedOutput")
        .def_readonly("output", &DecryptedOutput::output)
        .def_readonly("num_outputs", &DecryptedOutput::num_outputs);

    py::class_<InferenceClient>(m, "InferenceClient")
        .def(py::init([](const std::string& client_dir) {
                 py::gil_scoped_release release;
                 return InferenceClient(client_dir);
             }),
             py::arg("client_dir"))
        .def("setup", &InferenceClient::setup, py::call_guard<py::gil_scoped_release>())
        .def("release", &InferenceClient::release, py::call_guard<py::gil_scoped_release>())
        .def("load_full_context",
             [](InferenceClient& self, const py::bytes& full_bytes) {
                 Bytes vec = from_pybytes(full_bytes);
                 py::gil_scoped_release release;
                 self.load_full_context(vec);
             },
             py::arg("full_bytes"))
        .def("export_eval_context",
             [](const InferenceClient& self) {
                 Bytes data;
                 {
                     py::gil_scoped_release release;
                     data = self.export_eval_context();
                 }
                 return to_pybytes(data);
             })
        .def("export_full_context",
             [](const InferenceClient& self) {
                 Bytes data;
                 {
                     py::gil_scoped_release release;
                     data = self.export_full_context();
                 }
                 return to_pybytes(data);
             })
        .def(
            "encrypt",
            [](const InferenceClient& self, const std::map<std::string, std::string>& input_csvs) {
                std::map<std::string, Bytes> result;
                {
                    py::gil_scoped_release release;
                    result = self.encrypt(input_csvs);
                }
                return to_pybytes_dict(result);
            },
            py::arg("input_csvs"))
        .def(
            "decrypt",
            [](InferenceClient& self, const py::dict& encrypted_outputs) {
                auto outputs = from_pybytes_dict(encrypted_outputs);
                std::map<std::string, DecryptedOutput> result;
                {
                    py::gil_scoped_release release;
                    result = self.decrypt(outputs);
                }
                return result;
            },
            py::arg("encrypted_outputs"));
}
