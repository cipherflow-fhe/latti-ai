#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "data_trans_layer.h"

void send_public_context(DataTransmission& data_trans, CkksContext& context);
CkksContext recv_public_context(DataTransmission& data_trans);
void send_public_context_map(DataTransmission& data_trans,
                             std::map<std::string, std::unique_ptr<CkksContext>>& ckks_contexts);
std::map<std::string, std::unique_ptr<CkksContext>> recv_public_context_map(DataTransmission& data_trans);
void send_task_id(DataTransmission& data_trans, const std::string& task_id);
std::string recv_task_id(DataTransmission& data_trans);
void send_ct_compress(DataTransmission& data_trans, CkksCompressedCiphertext& ct, CkksContext& context);
void recv_ct_compress(DataTransmission& data_trans, CkksCompressedCiphertext& ct, bool is_truncated);
void send_ct_vec_compress(DataTransmission& data_trans,
                          std::vector<CkksCompressedCiphertext>& ct,
                          CkksContext& context);
void recv_ct_vec_compress(DataTransmission& data_trans, std::vector<CkksCompressedCiphertext>& ct);
std::vector<CkksCiphertext> recv_and_add_vec_compress(DataTransmission& data_trans,
                                                      std::vector<CkksCiphertext>& input,
                                                      CkksContext& context);
