#pragma once

#include <vector>
#include <cinttypes>
#include "../util.h"
#include "../data_structs/feature.h"

enum class MpcProtoType {
    end,
    enc_to_share,
    enc_to_share_for_multi_channel_pack,
    share_to_enc_for_multi_channel_pack,
    enc_to_share_0d,
    share_to_enc,
    share_to_enc_0d,
    share_2d_to_0d,
    relu,
    relu_0d,
    relu6,
    max_pool,
    avg_pool,
    distance,
    recovery_share,
    argmax,
    div,
    reciprocal,
    sqrt,
    softmax,
    simple_poly_relu
};

class MpcTaskMetaData {
public:
    std::vector<MpcProtoType> types;
    std::vector<std::vector<uint8_t>> data;

    void append(MpcProtoType type, std::vector<std::string> fmt, ...);

    Bytes serialize() const;
    void deserialize(const Bytes& bytes);
};

Bytes va_to_bytes(std::vector<std::string> fmt, ...);

void bytes_to_va(const Bytes& bytes, std::vector<std::string> fmt, ...);
