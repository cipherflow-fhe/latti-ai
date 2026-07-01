#pragma once

#include <map>
#include <memory>
#include <string>

#include "fhe_ops_lib/fhe_lib_v2.h"
#include "util.h"

using fhe_ops_lib::CkksBtpContext;
using fhe_ops_lib::CkksBtpParameter;
using fhe_ops_lib::CkksContext;
using fhe_ops_lib::CkksParameter;

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

Bytes serialize_ckks_bootstrampping_contexts(std::map<std::string, std::unique_ptr<CkksBtpContext>>& ckks_contexts);

std::map<std::string, std::unique_ptr<CkksContext>> deserialize_ckks_contexts(const Bytes& bytes);

std::map<std::string, std::unique_ptr<CkksContext>> deserialize_ckks_bootstrampping_contexts(const Bytes& bytes);
