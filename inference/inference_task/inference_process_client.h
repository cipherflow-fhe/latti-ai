#pragma once

#include <map>
#include <memory>
#include <string>

#include "fhe_mpc.h"
#include "mpc_task_meta_data.h"

using fhe_ops_lib::CkksBtpContext;
using fhe_ops_lib::CkksBtpParameter;

extern double time_relu_total;

Feature2DShare client_enc_to_share(std::map<std::string, std::unique_ptr<CkksContext>>& ckks_contexts,
                                   const Bytes& meta_data_bytes,
                                   CkksContext*& context_in,
                                   CkksContext*& context_out);

Feature2DShare client_enc_to_share_simple(std::map<std::string, std::unique_ptr<CkksContext>>& ckks_contexts,
                                          const Bytes& meta_data_bytes,
                                          CkksContext*& context_in,
                                          CkksContext*& context_out);

Feature2DShare client_enc_to_share_for_multi_channel_pack(
    std::map<std::string, std::unique_ptr<CkksContext>>& ckks_contexts,
    const Bytes& meta_data_bytes,
    CkksContext*& context_in,
    CkksContext*& context_out);

Feature2DShare client_enc_to_share_for_multi_channel_pack_simple(
    std::map<std::string, std::unique_ptr<CkksContext>>& ckks_contexts,
    const Bytes& meta_data_bytes,
    CkksContext*& context_in,
    CkksContext*& context_out);

Feature2DShare client_maxpool(const Bytes& meta_data_bytes, const Feature2DShare& x, double pt_range);

Array1DUint process(std::map<std::string, std::unique_ptr<CkksContext>>* ckks_contexts);

std::map<std::string, std::unique_ptr<CkksParameter>> init_parameters(const std::string& project_path);

void generate_context_map(const std::string& project_path,
                          std::map<std::string, std::unique_ptr<CkksContext>>& public_context_map,
                          std::map<std::string, std::unique_ptr<CkksContext>>& secret_context_map);

void generate_bootstrampping_context_map(
    const std::string& project_path,
    std::map<std::string, std::unique_ptr<CkksBtpContext>>& public_context_map,
    std::map<std::string, std::unique_ptr<CkksBtpContext>>& secret_context_map);

void generate_context_map_with_seed(const std::string& project_path,
                                    const Bytes& seed,
                                    std::map<std::string, std::unique_ptr<CkksContext>>& public_context_map,
                                    std::map<std::string, std::unique_ptr<CkksContext>>& secret_context_map);

CkksContext create_context();

Bytes serialize_ckks_contexts(std::map<std::string, std::unique_ptr<CkksContext>>& ckks_contexts);

Bytes serialize_ckks_bootstrampping_contexts(
    std::map<std::string, std::unique_ptr<CkksBtpContext>>& ckks_contexts);

std::map<std::string, std::unique_ptr<CkksContext>> deserialize_ckks_contexts(const Bytes& bytes);

std::map<std::string, std::unique_ptr<CkksContext>> deserialize_ckks_bootstrampping_contexts(const Bytes& bytes);
