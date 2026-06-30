#include "mpc_data_transmission.h"

#include <iostream>

#include "mpc/mpc_session.h"
#include "util.h"

using namespace std;
using fhe_ops_lib::Bytes;

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

MpcDataTransmission::MpcDataTransmission(DataTransmission data_trans) : data_trans_(data_trans) {}

DataTransmission current_mpc_data_transmission() {
    return ::mpc::data_transmission();
}

void send_mpc_metadata(const MpcTaskMetaData& meta_data) {
    send_mpc_metadata_bytes(meta_data.serialize());
}

void send_mpc_metadata_bytes(const vector<uint8_t>& meta_data_bytes) {
    DataTransmission data_trans = current_mpc_data_transmission();
    data_trans.send_bytes(meta_data_bytes);
}

void send_mpc_end() {
    MpcTaskMetaData end_meta;
    end_meta.append(MpcProtoType::end, {});
    send_mpc_metadata(end_meta);
}

MpcDataTransmission MpcDataTransmission::current() {
    return MpcDataTransmission(current_mpc_data_transmission());
}

DataTransmission& MpcDataTransmission::raw() {
    return data_trans_;
}

void MpcDataTransmission::send_string(const string& value) {
    size_t size = value.size();
    data_trans_.send_data(&size, sizeof(size));
    if (size > 0) {
        data_trans_.send_data(value.data(), size);
    }
}

string MpcDataTransmission::receive_string() {
    size_t size = 0;
    data_trans_.recv_data(&size, sizeof(size));
    string value(size, '\0');
    if (size > 0) {
        data_trans_.recv_data(value.data(), size);
    }
    return value;
}

void MpcDataTransmission::send_encrypted_map(const EncryptedBytesMap& values) {
    size_t size = values.size();
    data_trans_.send_data(&size, sizeof(size));
    for (const auto& [name, bytes] : values) {
        cout << "[Transport] Sending ciphertext [" << name << "], bytes=" << bytes.size() << endl;
        send_string(name);
        data_trans_.send_bytes(bytes);
    }
    data_trans_.flush();
}

EncryptedBytesMap MpcDataTransmission::receive_encrypted_map() {
    size_t size = 0;
    data_trans_.recv_data(&size, sizeof(size));
    EncryptedBytesMap values;
    for (size_t i = 0; i < size; i++) {
        string name = receive_string();
        values[name] = data_trans_.receive_bytes();
        cout << "[Transport] Received ciphertext [" << name << "], bytes=" << values[name].size() << endl;
    }
    return values;
}

void MpcDataTransmission::send_plaintext_map(const PlaintextMap& values) {
    size_t size = values.size();
    data_trans_.send_data(&size, sizeof(size));
    for (const auto& [name, data] : values) {
        send_string(name);
        size_t data_size = data.size();
        data_trans_.send_data(&data_size, sizeof(data_size));
        if (data_size > 0) {
            data_trans_.send_data(data.data(), data_size * sizeof(double));
        }
    }
    data_trans_.flush();
}

PlaintextMap MpcDataTransmission::receive_plaintext_map() {
    size_t size = 0;
    data_trans_.recv_data(&size, sizeof(size));
    PlaintextMap values;
    for (size_t i = 0; i < size; i++) {
        string name = receive_string();
        size_t data_size = 0;
        data_trans_.recv_data(&data_size, sizeof(data_size));
        vector<double> data(data_size);
        if (data_size > 0) {
            data_trans_.recv_data(data.data(), data_size * sizeof(double));
        }
        values[name] = move(data);
    }
    return values;
}

void MpcDataTransmission::send_dump_flag(bool enabled) {
    unsigned char dump_flag = enabled ? 1 : 0;
    data_trans_.send_data(&dump_flag, sizeof(dump_flag));
    data_trans_.flush();
}

bool MpcDataTransmission::receive_dump_flag() {
    unsigned char dump_flag = 0;
    data_trans_.recv_data(&dump_flag, sizeof(dump_flag));
    return dump_flag != 0;
}

void MpcDataTransmission::send_context_bytes(const vector<uint8_t>& context_bytes) {
    data_trans_.send_bytes(context_bytes);
    data_trans_.flush();
}

vector<uint8_t> MpcDataTransmission::receive_context_bytes() {
    return data_trans_.receive_bytes();
}

void MpcDataTransmission::send_public_context(CkksContext& context) {
    data_trans_.send_bytes(context.serialize());
}

CkksContext MpcDataTransmission::recv_public_context() {
    return CkksContext::deserialize(data_trans_.receive_bytes());
}

void MpcDataTransmission::send_public_context_map(map<string, unique_ptr<CkksContext>>& ckks_contexts) {
    uint32_t ctx_size = ckks_contexts.size();
    data_trans_.send_data(&ctx_size, sizeof(uint32_t));
    for (auto& context : ckks_contexts) {
        const string& key = context.first;
        vector<char> key_bytes(key.begin(), key.end());
        uint32_t data_size = key_bytes.size();
        data_trans_.send_data(&data_size, sizeof(uint32_t));
        data_trans_.send_data(key_bytes.data(), data_size);

        auto ctx = context.second.get();
        Bytes public_context_raw_data = ctx->serialize();
        uint64_t length = public_context_raw_data.size();
        data_trans_.send_data(&length, sizeof(uint64_t));
        data_trans_.send_data(public_context_raw_data.data(), length);
    }
}

map<string, unique_ptr<CkksContext>> MpcDataTransmission::recv_public_context_map() {
    map<string, unique_ptr<CkksContext>> ckks_contexts;
    uint32_t ctx_size;
    data_trans_.recv_data(&ctx_size, sizeof(uint32_t));
    for (int i = 0; i < ctx_size; i++) {
        uint32_t data_size = 0;
        data_trans_.recv_data(&data_size, sizeof(data_size));
        vector<char> bytes(data_size);
        data_trans_.recv_data(bytes.data(), data_size);
        string key(bytes.begin(), bytes.end());
        uint64_t length;
        data_trans_.recv_data(&length, sizeof(uint64_t));
        Bytes public_context_raw_data(length, 0);
        data_trans_.recv_data(public_context_raw_data.data(), length);
        auto deserialized_public_context = CkksContext::deserialize(public_context_raw_data);
        ckks_contexts[key] = make_unique<CkksContext>(move(deserialized_public_context));
    }
    return ckks_contexts;
}

void MpcDataTransmission::send_task_id(const string& task_id) {
    stringstream ss;
    ss_write_string(ss, task_id);
    vector<char> bytes(ss.str().size());
    ss.read(bytes.data(), ss.str().size());
    uint32_t data_size = bytes.size();
    data_trans_.send_data(&data_size, sizeof(uint32_t));
    data_trans_.send_data(bytes.data(), data_size);
}

string MpcDataTransmission::recv_task_id() {
    uint32_t data_size = 0;
    string task_id;
    data_trans_.recv_data(&data_size, sizeof(data_size));
    vector<char> bytes(data_size);
    data_trans_.recv_data(bytes.data(), data_size);
    stringstream ss;
    ss.write(bytes.data(), bytes.size());
    ss_read_string(ss, &task_id);
    return task_id;
}

void MpcDataTransmission::send_ct_compress(CkksCompressedCiphertext& ct, CkksContext& context) {
    data_trans_.send_bytes(ct.serialize(context.get_parameter()));
}

void MpcDataTransmission::recv_ct_compress(CkksCompressedCiphertext& ct, bool is_truncated) {
    (void)is_truncated;
    ct = CkksCompressedCiphertext::deserialize(data_trans_.receive_bytes());
}

void MpcDataTransmission::send_ct_vec_compress(vector<CkksCompressedCiphertext>& ct, CkksContext& context) {
    data_trans_.send_bytes(serialize_vector(context.get_parameter(), ct));
}

void MpcDataTransmission::recv_ct_vec_compress(vector<CkksCompressedCiphertext>& ct) {
    auto cts = deserialize_vector<CkksCompressedCiphertext>(data_trans_.receive_bytes());
    for (auto& c : cts) {
        ct.push_back(move(c));
    }
}

vector<CkksCiphertext> MpcDataTransmission::recv_and_add_vec_compress(vector<CkksCiphertext>& input,
                                                                      CkksContext& context) {
    vector<CkksCiphertext> res;
    vector<CkksCompressedCiphertext> recv_cipher;
    recv_ct_vec_compress(recv_cipher);
    printf("recv 208 ok\n");
    for (int i = 0; i < recv_cipher.size(); i++) {
        res.push_back(move(context.add(input[i], context.compressed_ciphertext_to_ciphertext(recv_cipher[i]))));
    }
    return res;
}
