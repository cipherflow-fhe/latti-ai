#include "mpc_data_transmission.h"

using namespace std;

namespace {
template <typename T>
Bytes serialize_vector(const CkksParameter& param, const vector<T>& objects) {
    stringstream ss;
    size_t n_object = objects.size();
    ss_write(ss, n_object);
    for (int i = 0; i < n_object; i++) {
        Bytes obj_bytes = objects[i].serialize(param);
        ss_write_vector(ss, obj_bytes);
    }
    return ss_to_bytes(ss);
}

template <typename T>
vector<T> deserialize_vector(const Bytes& bytes) {
    stringstream ss;
    bytes_to_ss(bytes, ss);
    size_t n_object;
    ss_read(ss, &n_object);
    vector<T> objects;
    for (int i = 0; i < n_object; i++) {
        Bytes obj_bytes;
        ss_read_vector(ss, &obj_bytes);
        objects.push_back(T::deserialize(obj_bytes));
    }
    return objects;
}
}

void send_public_context(DataTransmission& data_trans, CkksContext& context) {
    data_trans.send_bytes(context.serialize());
}

CkksContext recv_public_context(DataTransmission& data_trans) {
    return CkksContext::deserialize(data_trans.receive_bytes());
}

void send_public_context_map(DataTransmission& data_trans,
                             map<string, unique_ptr<CkksContext>>& ckks_contexts) {
    uint32_t ctx_size = ckks_contexts.size();
    data_trans.send_data(&ctx_size, sizeof(uint32_t));
    for (auto& context : ckks_contexts) {
        const string& key = context.first;
        vector<char> key_bytes(key.begin(), key.end());
        uint32_t data_size = key_bytes.size();
        data_trans.send_data(&data_size, sizeof(uint32_t));
        data_trans.send_data(key_bytes.data(), data_size);

        auto ctx = context.second.get();
        Bytes public_context_raw_data = ctx->serialize();
        uint64_t length = public_context_raw_data.size();
        data_trans.send_data(&length, sizeof(uint64_t));
        data_trans.send_data(public_context_raw_data.data(), length);
    }
}

map<string, unique_ptr<CkksContext>> recv_public_context_map(DataTransmission& data_trans) {
    map<string, unique_ptr<CkksContext>> ckks_contexts;
    uint32_t ctx_size;
    data_trans.recv_data(&ctx_size, sizeof(uint32_t));
    for (int i = 0; i < ctx_size; i++) {
        uint32_t data_size = 0;
        data_trans.recv_data(&data_size, sizeof(data_size));
        vector<char> bytes(data_size);
        data_trans.recv_data(bytes.data(), data_size);
        string key(bytes.begin(), bytes.end());
        uint64_t length;
        data_trans.recv_data(&length, sizeof(uint64_t));
        Bytes public_context_raw_data(length, 0);
        data_trans.recv_data(public_context_raw_data.data(), length);
        auto deserialized_public_context = CkksContext::deserialize(public_context_raw_data);
        ckks_contexts[key] = make_unique<CkksContext>(move(deserialized_public_context));
    }
    return ckks_contexts;
}

void send_task_id(DataTransmission& data_trans, const string& task_id) {
    stringstream ss;
    ss_write_string(ss, task_id);
    vector<char> bytes(ss.str().size());
    ss.read(bytes.data(), ss.str().size());
    uint32_t data_size = bytes.size();
    data_trans.send_data(&data_size, sizeof(uint32_t));
    data_trans.send_data(bytes.data(), data_size);
}

string recv_task_id(DataTransmission& data_trans) {
    uint32_t data_size = 0;
    string task_id;
    data_trans.recv_data(&data_size, sizeof(data_size));
    vector<char> bytes(data_size);
    data_trans.recv_data(bytes.data(), data_size);
    stringstream ss;
    ss.write(bytes.data(), bytes.size());
    ss_read_string(ss, &task_id);
    return task_id;
}

void send_ct_compress(DataTransmission& data_trans, CkksCompressedCiphertext& ct, CkksContext& context) {
    data_trans.send_bytes(ct.serialize(context.get_parameter()));
}

void recv_ct_compress(DataTransmission& data_trans, CkksCompressedCiphertext& ct, bool is_truncated) {
    (void)is_truncated;
    ct = CkksCompressedCiphertext::deserialize(data_trans.receive_bytes());
}

void send_ct_vec_compress(DataTransmission& data_trans,
                          vector<CkksCompressedCiphertext>& ct,
                          CkksContext& context) {
    data_trans.send_bytes(serialize_vector(context.get_parameter(), ct));
}

void recv_ct_vec_compress(DataTransmission& data_trans, vector<CkksCompressedCiphertext>& ct) {
    auto cts = deserialize_vector<CkksCompressedCiphertext>(data_trans.receive_bytes());
    for (auto& c : cts) {
        ct.push_back(move(c));
    }
}

vector<CkksCiphertext> recv_and_add_vec_compress(DataTransmission& data_trans,
                                                 vector<CkksCiphertext>& input,
                                                 CkksContext& context) {
    vector<CkksCiphertext> res;
    vector<CkksCompressedCiphertext> recv_cipher;
    recv_ct_vec_compress(data_trans, recv_cipher);
    printf("recv 208 ok\n");
    for (int i = 0; i < recv_cipher.size(); i++) {
        res.push_back(move(context.add(input[i], context.compressed_ciphertext_to_ciphertext(recv_cipher[i]))));
    }
    return res;
}
