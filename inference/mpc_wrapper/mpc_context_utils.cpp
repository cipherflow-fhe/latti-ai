#include "mpc_context_utils.h"

#include <iostream>
#include <sstream>

using namespace std;

map<string, unique_ptr<CkksParameter>> init_parameters(const string& project_path) {
    map<string, unique_ptr<CkksParameter>> ckks_parameters;
    auto json_params = read_json(project_path);
    for (auto& param : json_params.items()) {
        string key = param.key();
        ckks_parameters[key] = make_unique<CkksParameter>(CkksParameter::create_fpga_parameter());
    }
    return ckks_parameters;
}

void generate_context_map(const string& project_path,
                          map<string, unique_ptr<CkksContext>>& public_context_map,
                          map<string, unique_ptr<CkksContext>>& secret_context_map) {
    auto json_params = read_json(project_path);
    cout << json_params << endl;
    for (auto& json_param : json_params.items()) {
        const string& key = json_param.key();
        int N = json_param.value()["poly_modulus_degree"];
        CkksParameter param = CkksParameter::create_parameter(N);
        CkksContext context = CkksContext::create_random_context(param);
        context.gen_rotation_keys();
        auto ctx = context.make_public_context();
        public_context_map[key] = make_unique<CkksContext>(move(ctx));
        secret_context_map[key] = make_unique<CkksContext>(move(context));
    }
}

void generate_bootstrampping_context_map(const string& project_path,
                                         map<string, unique_ptr<CkksBtpContext>>& public_context_map,
                                         map<string, unique_ptr<CkksBtpContext>>& secret_context_map) {
    auto json_params = read_json(project_path);
    cout << json_params << endl;
    for (auto& json_param : json_params.items()) {
        const string& key = json_param.key();
        int N = json_param.value()["poly_modulus_degree"];
        CkksBtpParameter param = CkksBtpParameter::create_parameter();
        CkksBtpContext context = CkksBtpContext::create_random_context(param);
        context.gen_rotation_keys();
        auto ctx = context.make_public_context();
        public_context_map[key] = make_unique<CkksBtpContext>(move(ctx));
        secret_context_map[key] = make_unique<CkksBtpContext>(move(context));
    }
}

void generate_context_map_with_seed(const string& project_path,
                                    const Bytes& seed,
                                    map<string, unique_ptr<CkksContext>>& public_context_map,
                                    map<string, unique_ptr<CkksContext>>& secret_context_map) {
    (void)seed;
    auto json_params = read_json(project_path);
    for (auto& json_param : json_params.items()) {
        const string& key = json_param.key();
        int N = json_param.value()["poly_modulus_degree"];
        CkksParameter param = CkksParameter::create_parameter(N);
        cout << "N=" << N << endl;
        CkksContext context = CkksContext::create_random_context(param);
        context.gen_rotation_keys();
        auto ctx = context.make_public_context();
        public_context_map[key] = make_unique<CkksContext>(move(ctx));
        secret_context_map[key] = make_unique<CkksContext>(move(context));
    }
}

CkksContext create_context() {
    CkksParameter param = CkksParameter::create_fpga_parameter();
    CkksContext context = CkksContext::create_random_context(param);
    return context;
}

Bytes serialize_ckks_contexts(map<string, unique_ptr<CkksContext>>& ckks_contexts) {
    stringstream ss;
    uint16_t n_context = ckks_contexts.size();
    ss_write(ss, n_context);
    for (auto& pair : ckks_contexts) {
        const string& key = pair.first;
        ss_write_string(ss, key);
        Bytes context_bytes = pair.second->serialize();
        ss_write_vector(ss, context_bytes);
    }
    return ss_to_bytes(ss);
}

Bytes serialize_ckks_bootstrampping_contexts(map<string, unique_ptr<CkksBtpContext>>& ckks_contexts) {
    stringstream ss;
    uint16_t n_context = ckks_contexts.size();
    ss_write(ss, n_context);
    for (auto& pair : ckks_contexts) {
        const string& key = pair.first;
        ss_write_string(ss, key);
        Bytes context_bytes = pair.second->serialize();
        ss_write_vector(ss, context_bytes);
    }
    return ss_to_bytes(ss);
}

map<string, unique_ptr<CkksContext>> deserialize_ckks_contexts(const Bytes& bytes) {
    stringstream ss;
    bytes_to_ss(bytes, ss);
    map<string, unique_ptr<CkksContext>> ckks_contexts;
    uint16_t n_context;
    ss_read(ss, &n_context);
    for (int i = 0; i < n_context; i++) {
        string key;
        ss_read_string(ss, &key);
        Bytes ct_bytes;
        ss_read_vector(ss, &ct_bytes);
        CkksContext context = CkksContext::deserialize(ct_bytes);
        ckks_contexts[key] = make_unique<CkksContext>(move(context));
    }
    return ckks_contexts;
}

map<string, unique_ptr<CkksContext>> deserialize_ckks_bootstrampping_contexts(const Bytes& bytes) {
    stringstream ss;
    bytes_to_ss(bytes, ss);
    map<string, unique_ptr<CkksContext>> ckks_contexts;
    uint16_t n_context;
    ss_read(ss, &n_context);
    for (int i = 0; i < n_context; i++) {
        string key;
        ss_read_string(ss, &key);
        Bytes ct_bytes;
        ss_read_vector(ss, &ct_bytes);
        CkksBtpContext context = CkksBtpContext::deserialize(ct_bytes);
        ckks_contexts[key] = make_unique<CkksBtpContext>(move(context));
    }
    return ckks_contexts;
}
