#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mpc/data_trans_layer.h"
#include "mpc/mpc_task_meta_data.h"
#include "fhe_ops_lib/fhe_lib_v2.h"

using fhe_ops_lib::CkksCiphertext;
using fhe_ops_lib::CkksCompressedCiphertext;
using fhe_ops_lib::CkksContext;
using fhe_ops_lib::CkksParameter;

using EncryptedBytesMap = std::map<std::string, std::vector<uint8_t>>;
using PlaintextMap = std::map<std::string, std::vector<double>>;

DataTransmission current_mpc_data_transmission();
void send_mpc_metadata(const MpcTaskMetaData& meta_data);
void send_mpc_metadata_bytes(const std::vector<uint8_t>& meta_data_bytes);
void send_mpc_end();

class MpcDataTransmission {
public:
    explicit MpcDataTransmission(DataTransmission data_trans);

    static MpcDataTransmission current();

    DataTransmission& raw();

    void send_string(const std::string& value);
    std::string receive_string();
    void send_encrypted_map(const EncryptedBytesMap& values);
    EncryptedBytesMap receive_encrypted_map();
    void send_plaintext_map(const PlaintextMap& values);
    PlaintextMap receive_plaintext_map();
    void send_dump_flag(bool enabled);
    bool receive_dump_flag();
    void send_context_bytes(const std::vector<uint8_t>& context_bytes);
    std::vector<uint8_t> receive_context_bytes();
    void send_public_context(CkksContext& context);
    CkksContext recv_public_context();
    void send_public_context_map(std::map<std::string, std::unique_ptr<CkksContext>>& ckks_contexts);
    std::map<std::string, std::unique_ptr<CkksContext>> recv_public_context_map();
    void send_task_id(const std::string& task_id);
    std::string recv_task_id();
    void send_ct_compress(CkksCompressedCiphertext& ct, CkksContext& context);
    void recv_ct_compress(CkksCompressedCiphertext& ct, bool is_truncated);
    void send_ct_vec_compress(std::vector<CkksCompressedCiphertext>& ct, CkksContext& context);
    void recv_ct_vec_compress(std::vector<CkksCompressedCiphertext>& ct);
    std::vector<CkksCiphertext> recv_and_add_vec_compress(std::vector<CkksCiphertext>& input, CkksContext& context);

private:
    DataTransmission data_trans_;
};
