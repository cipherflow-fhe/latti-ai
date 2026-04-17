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
#include "interface/inference_server.h"

namespace py = pybind11;

// Convert Bytes (vector<uint8_t>) to Python bytes object
static py::bytes to_pybytes(const Bytes& b) {
    return py::bytes(reinterpret_cast<const char*>(b.data()), b.size());
}

// Convert map<string, Bytes> to dict[str, bytes]
static py::dict to_pybytes_dict(const std::map<std::string, Bytes>& m) {
    py::dict d;
    for (auto& [k, v] : m) {
        d[py::cast(k)] = to_pybytes(v);
    }
    return d;
}

PYBIND11_MODULE(latti_inference, m) {
    m.doc() = "Python bindings for latti-ai FHE inference engine";

    py::class_<DecryptedOutput>(m, "DecryptedOutput")
        .def_readonly("output", &DecryptedOutput::output)
        .def_readonly("num_outputs", &DecryptedOutput::num_outputs);

    py::class_<InferenceClient>(m, "InferenceClient")
        .def(py::init<const std::string&>(), py::arg("client_dir"))
        .def("setup", &InferenceClient::setup, py::call_guard<py::gil_scoped_release>())
        .def("export_eval_context",
             [](const InferenceClient& self) {
                 Bytes data;
                 {
                     py::gil_scoped_release release;
                     data = self.export_eval_context();
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
        .def("decrypt", &InferenceClient::decrypt, py::arg("encrypted_outputs"),
             py::call_guard<py::gil_scoped_release>());

    py::class_<InferenceServer>(m, "InferenceServer")
        .def(py::init<const std::string&, bool>(), py::arg("server_dir"), py::arg("use_gpu") = false)
        .def("import_eval_context", &InferenceServer::import_eval_context, py::arg("eval_context"),
             py::call_guard<py::gil_scoped_release>())
        .def("load_model", &InferenceServer::load_model, py::call_guard<py::gil_scoped_release>())
        .def(
            "evaluate",
            [](InferenceServer& self, const std::map<std::string, Bytes>& encrypted_inputs) {
                std::map<std::string, Bytes> result;
                {
                    py::gil_scoped_release release;
                    result = self.evaluate(encrypted_inputs);
                }
                return to_pybytes_dict(result);
            },
            py::arg("encrypted_inputs"))
        .def("evaluate_plaintext", &InferenceServer::evaluate_plaintext, py::arg("input_csvs"),
             py::call_guard<py::gil_scoped_release>());
}
