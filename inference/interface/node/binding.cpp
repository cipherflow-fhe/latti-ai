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

#include <napi.h>

#include "interface/inference_client.h"

// ── Helpers ──────────────────────────────────────────────────────────────────

static Napi::Buffer<uint8_t> bytes_to_buffer(Napi::Env env, const Bytes& b) {
    return Napi::Buffer<uint8_t>::Copy(env, b.data(), b.size());
}

static Bytes buffer_to_bytes(Napi::Value val) {
    auto buf = val.As<Napi::Buffer<uint8_t>>();
    return Bytes(buf.Data(), buf.Data() + buf.Length());
}

// ── AsyncWorker wrappers ─────────────────────────────────────────────────────

class SetupWorker : public Napi::AsyncWorker {
public:
    SetupWorker(Napi::Env env, InferenceClient* client)
        : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)), client_(client) {}

    Napi::Promise Promise() {
        return deferred_.Promise();
    }

    void Execute() override {
        try {
            client_->setup();
        } catch (const std::exception& e) { SetError(e.what()); }
    }

    void OnOK() override {
        deferred_.Resolve(Env().Undefined());
    }
    void OnError(const Napi::Error& err) override {
        deferred_.Reject(err.Value());
    }

private:
    Napi::Promise::Deferred deferred_;
    InferenceClient* client_;
};

class ExportEvalContextWorker : public Napi::AsyncWorker {
public:
    ExportEvalContextWorker(Napi::Env env, InferenceClient* client)
        : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)), client_(client) {}

    Napi::Promise Promise() {
        return deferred_.Promise();
    }

    void Execute() override {
        try {
            result_ = client_->export_eval_context();
        } catch (const std::exception& e) { SetError(e.what()); }
    }

    void OnOK() override {
        deferred_.Resolve(bytes_to_buffer(Env(), result_));
    }
    void OnError(const Napi::Error& err) override {
        deferred_.Reject(err.Value());
    }

private:
    Napi::Promise::Deferred deferred_;
    InferenceClient* client_;
    Bytes result_;
};

class EncryptWorker : public Napi::AsyncWorker {
public:
    EncryptWorker(Napi::Env env, InferenceClient* client, std::map<std::string, std::string> input_csvs)
        : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)), client_(client),
          input_csvs_(std::move(input_csvs)) {}

    Napi::Promise Promise() {
        return deferred_.Promise();
    }

    void Execute() override {
        try {
            result_ = client_->encrypt(input_csvs_);
        } catch (const std::exception& e) { SetError(e.what()); }
    }

    void OnOK() override {
        auto env = Env();
        auto obj = Napi::Object::New(env);
        for (auto& [k, v] : result_) {
            obj.Set(k, bytes_to_buffer(env, v));
        }
        deferred_.Resolve(obj);
    }

    void OnError(const Napi::Error& err) override {
        deferred_.Reject(err.Value());
    }

private:
    Napi::Promise::Deferred deferred_;
    InferenceClient* client_;
    std::map<std::string, std::string> input_csvs_;
    std::map<std::string, Bytes> result_;
};

class DecryptWorker : public Napi::AsyncWorker {
public:
    DecryptWorker(Napi::Env env, InferenceClient* client, std::map<std::string, Bytes> encrypted)
        : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)), client_(client),
          encrypted_(std::move(encrypted)) {}

    Napi::Promise Promise() {
        return deferred_.Promise();
    }

    void Execute() override {
        try {
            result_ = client_->decrypt(encrypted_);
        } catch (const std::exception& e) { SetError(e.what()); }
    }

    void OnOK() override {
        auto env = Env();
        auto obj = Napi::Object::New(env);
        for (auto& [k, v] : result_) {
            auto entry = Napi::Object::New(env);
            auto arr = Napi::Float64Array::New(env, v.output.size());
            for (size_t i = 0; i < v.output.size(); ++i)
                arr[i] = v.output[i];
            entry.Set("output", arr);
            entry.Set("numOutputs", Napi::Number::New(env, v.num_outputs));
            obj.Set(k, entry);
        }
        deferred_.Resolve(obj);
    }

    void OnError(const Napi::Error& err) override {
        deferred_.Reject(err.Value());
    }

private:
    Napi::Promise::Deferred deferred_;
    InferenceClient* client_;
    std::map<std::string, Bytes> encrypted_;
    std::map<std::string, DecryptedOutput> result_;
};

class ExportFullContextWorker : public Napi::AsyncWorker {
public:
    ExportFullContextWorker(Napi::Env env, InferenceClient* client)
        : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)), client_(client) {}

    Napi::Promise Promise() {
        return deferred_.Promise();
    }

    void Execute() override {
        try {
            result_ = client_->export_full_context();
        } catch (const std::exception& e) { SetError(e.what()); }
    }

    void OnOK() override {
        deferred_.Resolve(bytes_to_buffer(Env(), result_));
    }
    void OnError(const Napi::Error& err) override {
        deferred_.Reject(err.Value());
    }

private:
    Napi::Promise::Deferred deferred_;
    InferenceClient* client_;
    Bytes result_;
};

class LoadFullContextWorker : public Napi::AsyncWorker {
public:
    LoadFullContextWorker(Napi::Env env, InferenceClient* client, Bytes full_bytes)
        : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)), client_(client),
          full_bytes_(std::move(full_bytes)) {}

    Napi::Promise Promise() {
        return deferred_.Promise();
    }

    void Execute() override {
        try {
            client_->load_full_context(full_bytes_);
        } catch (const std::exception& e) { SetError(e.what()); }
    }

    void OnOK() override {
        deferred_.Resolve(Env().Undefined());
    }
    void OnError(const Napi::Error& err) override {
        deferred_.Reject(err.Value());
    }

private:
    Napi::Promise::Deferred deferred_;
    InferenceClient* client_;
    Bytes full_bytes_;
};

// ── InferenceClient wrapper class ────────────────────────────────────────────

class InferenceClientWrapper : public Napi::ObjectWrap<InferenceClientWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func =
            DefineClass(env, "InferenceClient",
                        {
                            InstanceMethod<&InferenceClientWrapper::Setup>("setup"),
                            InstanceMethod<&InferenceClientWrapper::ExportEvalContext>("exportEvalContext"),
                            InstanceMethod<&InferenceClientWrapper::ExportFullContext>("exportFullContext"),
                            InstanceMethod<&InferenceClientWrapper::LoadFullContext>("loadFullContext"),
                            InstanceMethod<&InferenceClientWrapper::Encrypt>("encrypt"),
                            InstanceMethod<&InferenceClientWrapper::Decrypt>("decrypt"),
                        });

        auto* constructor = new Napi::FunctionReference();
        *constructor = Napi::Persistent(func);
        exports.Set("InferenceClient", func);
        env.SetInstanceData(constructor);
        return exports;
    }

    InferenceClientWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<InferenceClientWrapper>(info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsString()) {
            Napi::TypeError::New(env, "Expected string argument: client_dir").ThrowAsJavaScriptException();
            return;
        }
        std::string client_dir = info[0].As<Napi::String>().Utf8Value();
        client_ = std::make_unique<InferenceClient>(client_dir);
    }

    Napi::Value Setup(const Napi::CallbackInfo& info) {
        auto* worker = new SetupWorker(info.Env(), client_.get());
        worker->Queue();
        return worker->Promise();
    }

    Napi::Value ExportEvalContext(const Napi::CallbackInfo& info) {
        auto* worker = new ExportEvalContextWorker(info.Env(), client_.get());
        worker->Queue();
        return worker->Promise();
    }

    Napi::Value Encrypt(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsObject()) {
            Napi::TypeError::New(env, "Expected object argument: {inputName: csvPath}").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        auto obj = info[0].As<Napi::Object>();
        std::map<std::string, std::string> input_csvs;
        auto names = obj.GetPropertyNames();
        for (uint32_t i = 0; i < names.Length(); i++) {
            auto key = names.Get(i).As<Napi::String>().Utf8Value();
            auto val = obj.Get(key).As<Napi::String>().Utf8Value();
            input_csvs[key] = val;
        }

        auto* worker = new EncryptWorker(env, client_.get(), std::move(input_csvs));
        worker->Queue();
        return worker->Promise();
    }

    Napi::Value Decrypt(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsObject()) {
            Napi::TypeError::New(env, "Expected object argument: {outputName: Buffer}").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        auto obj = info[0].As<Napi::Object>();
        std::map<std::string, Bytes> encrypted;
        auto names = obj.GetPropertyNames();
        for (uint32_t i = 0; i < names.Length(); i++) {
            auto key = names.Get(i).As<Napi::String>().Utf8Value();
            encrypted[key] = buffer_to_bytes(obj.Get(key));
        }

        auto* worker = new DecryptWorker(env, client_.get(), std::move(encrypted));
        worker->Queue();
        return worker->Promise();
    }

    Napi::Value ExportFullContext(const Napi::CallbackInfo& info) {
        auto* w = new ExportFullContextWorker(info.Env(), client_.get());
        w->Queue();
        return w->Promise();
    }

    Napi::Value LoadFullContext(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsBuffer()) {
            Napi::TypeError::New(env, "Expected Buffer argument: full_bytes").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        auto* w = new LoadFullContextWorker(env, client_.get(), buffer_to_bytes(info[0]));
        w->Queue();
        return w->Promise();
    }

private:
    std::unique_ptr<InferenceClient> client_;
};

// ── Module init ──────────────────────────────────────────────────────────────

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    InferenceClientWrapper::Init(env, exports);
    return exports;
}

NODE_API_MODULE(latti_client, InitAll)
