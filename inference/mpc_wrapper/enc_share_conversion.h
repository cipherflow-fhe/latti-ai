#pragma once

#include <map>
#include <memory>

#include "inference_task/inference_process.h"
#include "mpc/mpc_numeric.h"
#include "mpc/mpc_task_meta_data.h"

using fhe_ops_lib::CkksContext;

class EncToShareClient {
public:
    EncToShareClient(std::map<std::string, std::unique_ptr<CkksContext>>& ckks_contexts,
                     CkksContext*& context_in,
                     CkksContext*& context_out,
                     int scale_ord = mpc::DEFAULT_SCALE_BIT,
                     uint64_t ring_mod = mpc::RING_MOD);

    Feature2DShare client_enc_to_share(const Bytes& meta_data_bytes);

    Feature2DShare client_enc_to_share_simple(const Bytes& meta_data_bytes);

    Feature2DShare client_enc_to_share_for_multi_channel_pack(const Bytes& meta_data_bytes);

    Feature2DShare client_enc_to_share_for_multi_channel_pack_simple(const Bytes& meta_data_bytes);

    Feature0DShare client_enc_to_share_0d(const Bytes& meta_data_bytes);

    Feature2DShare decrypt_to_share(const Feature2DEncrypted& x_enc,
                                    PackType pack_type = PackType::MultiplexedPacking) const;

    Feature2DShare decrypt_to_share_simple(const Feature2DEncrypted& x_enc,
                                           PackType pack_type = PackType::MultipleChannelPacking) const;

    Feature0DShare decrypt_to_share(const Feature0DEncrypted& x_enc, int n_channel) const;

private:
    std::map<std::string, std::unique_ptr<CkksContext>>& ckks_contexts_;
    CkksContext*& context_in_;
    CkksContext*& context_out_;
    int scale_ord_;
    uint64_t ring_mod_;
};

class ShareToEncClient {
public:
    ShareToEncClient(CkksContext& context_out,
                     int scale_ord = mpc::DEFAULT_SCALE_BIT,
                     uint64_t ring_mod = mpc::RING_MOD,
                     double pt_range = 128.0);

    void client_share_to_enc(Feature2DShare& share, const Bytes& meta_data_bytes);

    void client_share_to_enc_simple(Feature2DShare& share, const Bytes& meta_data_bytes);

    void client_share_to_enc_for_multi_channel_pack(Feature2DShare& share, const Bytes& meta_data_bytes);

    void client_share_to_enc_for_multi_channel_pack_simple(Feature2DShare& share, const Bytes& meta_data_bytes);

    void client_share_to_enc_0d(Feature0DShare& share, const Bytes& meta_data_bytes);

    Array<uint64_t, 1> encrypt_from_share(Feature2DEncrypted& x_enc,
                                          const Feature2DShare& share,
                                          int n_channel,
                                          const Duo& input_shape,
                                          PackType pack_type = PackType::MultiplexedPacking) const;

    void encrypt_from_share_simple(Feature2DEncrypted& x_enc,
                                   const Feature2DShare& share,
                                   int n_channel,
                                   const Duo& input_shape,
                                   PackType pack_type = PackType::MultipleChannelPacking,
                                   bool use_recode = false) const;

    Array<uint64_t, 1> encrypt_from_share(Feature0DEncrypted& x_enc,
                                          const Feature0DShare& share,
                                          int n_channel) const;

private:
    CkksContext& context_out_;
    int scale_ord_;
    uint64_t ring_mod_;
    double pt_range_;
};

class EncToShareServer {
public:
    EncToShareServer(CkksContext& context, int scale_ord = mpc::DEFAULT_SCALE_BIT, uint64_t ring_mod = mpc::RING_MOD);

    Feature2DShare server_enc_to_share_multi_pack(const Feature2DEncrypted& x_enc,
                                                  PackType pack_type = PackType::MultiplexedPacking);

    Feature2DShare server_enc_to_share_multi_pack_simple(const Feature2DEncrypted& x_enc,
                                                         PackType pack_type = PackType::MultiplexedPacking);

    Feature2DShare server_enc_to_share(const Feature2DEncrypted& x_enc);

    Feature2DShare server_enc_to_share_simple(const Feature2DEncrypted& x_enc);

    Feature0DShare server_enc_to_share(const Feature0DEncrypted& x_enc);

    void split_to_shares(const Feature2DEncrypted& x_enc, Feature2DEncrypted* share0, Feature2DShare* share1) const;

    void split_to_shares_simple(const Feature2DEncrypted& x_enc,
                                Feature2DEncrypted* share0,
                                Feature2DShare* share1) const;

    void split_to_shares_for_multi_channel_pack(
        const Feature2DEncrypted& x_enc,
        Feature2DEncrypted* share0,
        Feature2DShare* share1,
        PackType pack_type = PackType::MultiplexedPacking) const;

    void split_to_shares_for_multi_channel_pack_simple(
        const Feature2DEncrypted& x_enc,
        Feature2DEncrypted* share0,
        Feature2DShare* share1,
        PackType pack_type = PackType::MultiplexedPacking) const;

    void split_to_shares(const Feature0DEncrypted& x_enc, Feature0DEncrypted* share0, Feature0DShare* share1) const;

    void split_to_shares_reshape(const Feature0DEncrypted& x_enc,
                                 Feature0DEncrypted* share0,
                                 Feature0DShare* share1) const;

private:
    CkksContext& context_;
    int scale_ord_;
    uint64_t ring_mod_;
};

class ShareToEncServer {
public:
    ShareToEncServer(CkksContext& context,
                     int scale_ord = mpc::DEFAULT_SCALE_BIT,
                     uint64_t ring_mod = mpc::RING_MOD,
                     double pt_range = 128.0);

    Feature2DEncrypted server_share_to_enc_multi_pack(Feature2DShare& y_share0,
                                                      int level,
                                                      PackType pack_type = PackType::MultiplexedPacking);

    Feature2DEncrypted server_share_to_enc_multi_pack_simple(Feature2DShare& y_share0,
                                                             int level,
                                                             PackType pack_type = PackType::MultiplexedPacking);

    Feature2DEncrypted server_share_to_enc_simple(Feature2DShare& y_share0, int level);

    Feature0DEncrypted share_to_enc(Feature0DShare& y_share0, int level);

    Feature2DEncrypted combine_with_share(const Feature2DEncrypted& y_share1_enc,
                                          const Feature2DShare& share) const;

    Feature2DEncrypted combine_with_share_simple(const Feature2DEncrypted& y_share1_enc,
                                                 const Feature2DShare& share) const;

    Feature2DEncrypted combine_with_share_simple_for_multi_pack(
        const Feature2DEncrypted& y_share1_enc,
        const Feature2DShare& share,
        PackType pack_type = PackType::MultiplexedPacking) const;

    Feature2DEncrypted combine_with_share_new_protocol(const Feature2DEncrypted& y_share1_enc,
                                                       const Feature2DShare& share,
                                                       const Feature2DEncrypted& y_share2_enc,
                                                       const Bytes& b1) const;

    Feature2DEncrypted combine_with_share_new_protocol_for_multi_pack(
        const Feature2DEncrypted& y_share1_enc,
        const Feature2DShare& share,
        const Feature2DEncrypted& y_share2_enc,
        const Bytes& b1,
        PackType pack_type = PackType::MultiplexedPacking) const;

    Feature0DEncrypted combine_with_share(const Feature0DEncrypted& y_share1_enc,
                                          const Feature0DShare& share) const;

    Feature0DEncrypted combine_with_share_new_protocol(const Feature0DEncrypted& y_share1_enc,
                                                       const Feature0DShare& share,
                                                       const Feature0DEncrypted& y_share2_enc,
                                                       const Bytes& b1) const;

private:
    CkksContext& context_;
    int scale_ord_;
    uint64_t ring_mod_;
    double pt_range_;
};
