#include "inference_process_client.h"
#include "fhe-mpc/mpc/SCI/src/globals.h"

using namespace std;

double time_relu_total = 0;

Feature2DShare client_enc_to_share(map<string, unique_ptr<CkksContext>>& ckks_contexts,
                                   const Bytes& meta_data_bytes,
                                   CkksContext*& context_in,
                                   CkksContext*& context_out) {
    DataTransmission data_trans(io);

    uint8_t level;
    uint8_t param_id_in;
    uint8_t param_id_out;
    bytes_to_va(meta_data_bytes, {"u8", "u8", "u8"}, &level, &param_id_in, &param_id_out);

    string param_in = param_to_string(param_id_in);
    string param_out = param_to_string(param_id_out);
    context_in = ckks_contexts[param_in].get();
    context_out = ckks_contexts[param_out].get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    Feature2DEncrypted x_e(context_in, level);
    x_e.deserialize(x_e_bytes);
    Feature2DShare y = Feature2DShare(RING_MOD, DEFAULT_SCALE_BIT);
    x_e.decrypt_to_share(&y, PackType::MultipleChannelPacking);

    return y;
}

Feature2DShare client_enc_to_share_simple(map<string, unique_ptr<CkksContext>>& ckks_contexts,
                                          const Bytes& meta_data_bytes,
                                          CkksContext*& context_in,
                                          CkksContext*& context_out) {
    DataTransmission data_trans(io);

    uint8_t level;
    uint8_t param_id_in;
    uint8_t param_id_out;
    bytes_to_va(meta_data_bytes, {"u8", "u8", "u8"}, &level, &param_id_in, &param_id_out);

    string param_in = param_to_string(param_id_in);
    string param_out = param_to_string(param_id_out);
    context_in = ckks_contexts[param_in].get();
    context_out = ckks_contexts[param_out].get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    Feature2DEncrypted x_e(context_in, level);
    x_e.deserialize(x_e_bytes);
    Feature2DShare y = Feature2DShare(RING_MOD, DEFAULT_SCALE_BIT);
    x_e.decrypt_to_share_simple(&y, PackType::MultipleChannelPacking);

    return y;
}

Feature2DShare client_enc_to_share_for_multi_channel_pack(map<string, unique_ptr<CkksContext>>& ckks_contexts,
                                                          const Bytes& meta_data_bytes,
                                                          CkksContext*& context_in,
                                                          CkksContext*& context_out) {
    DataTransmission data_trans(io);
    uint8_t level;
    uint8_t param_id_in;
    uint8_t param_id_out;
    uint8_t temp_int = 0;
    bytes_to_va(meta_data_bytes, {"u8", "u8", "u8", "u8"}, &level, &param_id_in, &param_id_out, &temp_int);
    PackType pack_type = (PackType)temp_int;

    string param_in = param_to_string(param_id_in);
    string param_out = param_to_string(param_id_out);
    context_in = ckks_contexts[param_in].get();
    context_out = ckks_contexts[param_out].get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    Feature2DEncrypted x_e(context_in, level);
    x_e.deserialize(x_e_bytes);
    Feature2DShare y = Feature2DShare(RING_MOD, DEFAULT_SCALE_BIT);
    x_e.decrypt_to_share(&y, pack_type);

    return y;
}

Feature2DShare client_enc_to_share_for_multi_channel_pack_simple(map<string, unique_ptr<CkksContext>>& ckks_contexts,
                                                                 const Bytes& meta_data_bytes,
                                                                 CkksContext*& context_in,
                                                                 CkksContext*& context_out) {
    DataTransmission data_trans(io);
    uint8_t level;
    uint8_t param_id_in;
    uint8_t param_id_out;
    uint8_t temp_int = 0;
    bytes_to_va(meta_data_bytes, {"u8", "u8", "u8", "u8"}, &level, &param_id_in, &param_id_out, &temp_int);
    PackType pack_type = (PackType)temp_int;

    string param_in = param_to_string(param_id_in);
    string param_out = param_to_string(param_id_out);
    context_in = ckks_contexts[param_in].get();
    context_out = ckks_contexts[param_out].get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    Feature2DEncrypted x_e(context_in, level);
    x_e.deserialize(x_e_bytes);
    Feature2DShare y = Feature2DShare(RING_MOD, DEFAULT_SCALE_BIT);
    x_e.decrypt_to_share_simple(&y, pack_type);

    return y;
}

Feature2DShare client_maxpool(const Bytes& meta_data_bytes, const Feature2DShare& x, double pt_range) {
    Duo kernel_shape;
    Duo pool_stride;
    bytes_to_va(meta_data_bytes, {"duo", "duo"}, &kernel_shape, &pool_stride);

    PoolLayerClient pool(kernel_shape, pool_stride, DEFAULT_SCALE_BIT, RING_MOD, MAXPOOL, pt_range);
    Feature2DShare y(x.ring_mod, x.scale_ord);
    pool.run(x, y);

    return y;
}

Array1DUint process(map<string, unique_ptr<CkksContext>>* ckks_contexts) {
    DataTransmission data_trans(io);
    int scale_ord = DEFAULT_SCALE_BIT;
    double pt_range = 128.0;
    uint64_t ring_mod = RING_MOD;

    unique_ptr<Feature2DShare> im_2d;
    unique_ptr<Feature0DShare> im_0d;

    while (true) {
        Bytes meta_data_bytes = data_trans.receive_bytes();
        cout << "recv meta ok" << endl;
        MpcTaskMetaData meta_data;
        meta_data.deserialize(meta_data_bytes);

        size_t n_proto = meta_data.types.size();
        CkksContext* context_in;
        CkksContext* context_out;

        for (int i = 0; i < n_proto; i++) {
            MpcProtoType type = meta_data.types[i];
            if (type == MpcProtoType::end) {
                if (im_0d == nullptr) {
                    Array1DUint temp;
                    return temp;
                }
                return im_0d->data.to_array_1d();
            } else if (type == MpcProtoType::enc_to_share || type == MpcProtoType::enc_to_share_simple) {
                if (type == MpcProtoType::enc_to_share_simple) {
                    im_2d = make_unique<Feature2DShare>(
                        client_enc_to_share_simple(*ckks_contexts, meta_data.data[i], context_in, context_out));
                } else {
                    im_2d = make_unique<Feature2DShare>(
                        client_enc_to_share(*ckks_contexts, meta_data.data[i], context_in, context_out));
                }

            } else if (type == MpcProtoType::enc_to_share_for_multi_channel_pack) {
                im_2d = make_unique<Feature2DShare>(client_enc_to_share_for_multi_channel_pack(
                    *ckks_contexts, meta_data.data[i], context_in, context_out));
            } else if (type == MpcProtoType::enc_to_share_for_multi_channel_pack_simple) {
                im_2d = make_unique<Feature2DShare>(client_enc_to_share_for_multi_channel_pack_simple(
                    *ckks_contexts, meta_data.data[i], context_in, context_out));
            } else if (type == MpcProtoType::enc_to_share_0d) {
                uint8_t level;
                uint32_t n_channal;
                uint8_t param_in_id;
                uint8_t param_out_id;

                bytes_to_va(meta_data.data[i], {"u8", "u32", "u8", "u8"}, &level, &n_channal, &param_in_id,
                            &param_out_id);
                string param_in = param_to_string(param_in_id);
                string param_out = param_to_string(param_out_id);
                context_in = ckks_contexts->at(param_in).get();
                context_out = ckks_contexts->at(param_out).get();

                Bytes x_e_bytes = data_trans.receive_bytes();
                if (context_in == nullptr) {
                    cout << "wrong ptr" << endl;
                }
                Feature0DEncrypted x_e(context_in, level);
                x_e.deserialize(x_e_bytes);
                im_0d = make_unique<Feature0DShare>(ring_mod, scale_ord);
                x_e.decrypt_to_share(im_0d.get(), n_channal);
            } else if (type == MpcProtoType::share_to_enc) {
                uint8_t level;
                uint32_t n_channel;
                bytes_to_va(meta_data.data[i], {"u8", "u32"}, &level, &n_channel);

                Feature2DEncrypted x_e(context_out, level, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
                for (int i = 0; i < im_2d->data.get_size(); i++) {
                    im_2d->data.set(i, im_2d->data.get(i) * T_SCALE % RING_MOD);
                }
                auto data_process =
                    x_e.encrypt_from_share(*im_2d, n_channel, im_2d->shape, PackType::MultipleChannelPacking);

                MPC mpc(scale_ord, ring_mod, pt_range);
                auto b0 = mpc.wrap_protocol(data_process.to_array_1d(), data_trans.io_in, otpack, party);

                Array<double, 1> b0_mult_mod_div_s_mg(im_2d->data.get_shape());
                double scale = DEFAULT_SCALE;
                for (int i = 0; i < b0.size(); i++) {
                    double temp_res = double(b0[i] * ring_mod) / scale;
                    b0_mult_mod_div_s_mg.set(i, temp_res);
                }
                CkksContext& ctx_extra = context_out->get_extra_level_context();
                Feature2DEncrypted send_ct(&ctx_extra, level + 1, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
                send_ct.pack_multiple_channel(
                    b0_mult_mod_div_s_mg.reshape<3>(
                        {(uint64_t)n_channel, (uint64_t)im_2d->shape[0], (uint64_t)im_2d->shape[1]}),
                    false, DEFAULT_SCALE);
                Bytes b = x_e.serialize();
                data_trans.send_bytes(b);
                Bytes send_bytes = send_ct.serialize();
                data_trans.send_bytes(send_bytes);
            } else if (type == MpcProtoType::share_to_enc_simple) {
                uint8_t level;
                uint32_t n_channel;
                bytes_to_va(meta_data.data[i], {"u8", "u32"}, &level, &n_channel);

                Feature2DEncrypted x_e(context_out, level, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
                x_e.encrypt_from_share_simple(*im_2d, n_channel, im_2d->shape, PackType::MultipleChannelPacking,
                                             MPC_REFRESH_USE_RECODE);

                Bytes b = x_e.serialize();
                data_trans.send_bytes(b);
            } else if (type == MpcProtoType::share_to_enc_for_multi_channel_pack) {
                uint8_t level;
                uint32_t n_channel;
                Duo skip;

                uint8_t temp_int = 0;
                bytes_to_va(meta_data.data[i], {"u8", "u32", "duo", "u8"}, &level, &n_channel, &skip, &temp_int);
                PackType pack_type = (PackType)temp_int;
                Feature2DEncrypted x_e(context_out, level, skip);
                for (int i = 0; i < im_2d->data.get_size(); i++) {
                    im_2d->data.set(i, im_2d->data.get(i) * T_SCALE % RING_MOD);
                }
                auto data_process = x_e.encrypt_from_share(*im_2d, n_channel, im_2d->shape, pack_type);

                MPC mpc(scale_ord, ring_mod, pt_range);
                auto b0 = mpc.wrap_protocol(data_process.to_array_1d(), data_trans.io_in, otpack, party);

                Array<double, 1> b0_mult_mod_div_s_mg(im_2d->data.get_shape());
                double scale = DEFAULT_SCALE;
                for (int i = 0; i < b0.size(); i++) {
                    double temp_res = double(b0[i] * ring_mod) / scale;
                    b0_mult_mod_div_s_mg.set(i, temp_res);
                }
                CkksContext& ctx_extra = context_out->get_extra_level_context();
                Feature2DEncrypted send_ct(&ctx_extra, level + 1, skip, {1, 1}, pack_type);
                auto send_mg = b0_mult_mod_div_s_mg.reshape<3>(
                    {(uint64_t)n_channel, (uint64_t)im_2d->shape[0], (uint64_t)im_2d->shape[1]});
                if (pack_type == PackType::MultipleChannelPacking) {
                    send_ct.pack_multiple_channel(send_mg, false, DEFAULT_SCALE);
                } else if (pack_type == PackType::MultiplexedPacking) {
                    send_ct.pack_multiplexed(send_mg, false, DEFAULT_SCALE);
                } else if (pack_type == PackType::InterleavedPacking) {
                    Duo block_expansion = {(uint32_t)ceil(im_2d->shape[0] / (double)BLOCK_SHAPE[0]),
                                           (uint32_t)ceil(im_2d->shape[1] / (double)BLOCK_SHAPE[1])};
                    send_ct.pack_interleaved(send_mg, BLOCK_SHAPE, block_expansion, false, DEFAULT_SCALE);
                }
                Bytes b = x_e.serialize();
                data_trans.send_bytes(b);
                Bytes send_bytes = send_ct.serialize();
                data_trans.send_bytes(send_bytes);
            } else if (type == MpcProtoType::share_to_enc_for_multi_channel_pack_simple) {
                uint8_t level;
                uint32_t n_channel;
                Duo skip;

                uint8_t temp_int = 0;
                bytes_to_va(meta_data.data[i], {"u8", "u32", "duo", "u8"}, &level, &n_channel, &skip, &temp_int);
                PackType pack_type = (PackType)temp_int;
                Feature2DEncrypted x_e(context_out, level, skip, {1, 1}, pack_type);
                x_e.encrypt_from_share_simple(*im_2d, n_channel, im_2d->shape, pack_type, MPC_REFRESH_USE_RECODE);

                Bytes b = x_e.serialize();
                data_trans.send_bytes(b);
            } else if (type == MpcProtoType::share_2d_to_0d) {
                im_0d = make_unique<Feature0DShare>(ring_mod, scale_ord);
                im_0d->data = std::move(im_2d->data);
            } else if (type == MpcProtoType::share_to_enc_0d) {
                uint8_t level = 0;
                uint32_t n_channel = 0;
                bytes_to_va(meta_data.data[i], {"u8", "u32"}, &level, &n_channel);
                Feature0DEncrypted x_e(context_out, level);
                x_e.skip = 1;
                for (int i = 0; i < im_0d->data.get_size(); i++) {
                    im_0d->data.set(i, im_0d->data.get(i) * T_SCALE % RING_MOD);
                }
                auto data_process = x_e.encrypt_from_share(*im_0d, n_channel);
                MPC mpc(scale_ord, ring_mod, pt_range);
                data_trans.io_in->flush();
                auto b0 = mpc.wrap_protocol(data_process.to_array_1d(), data_trans.io_in, otpack, party);
                data_trans.io_in->flush();

                Array<double, 1> send_mg(im_0d->data.get_shape());
                double scale = DEFAULT_SCALE;
                for (int i = 0; i < b0.size(); i++) {
                    double temp_res = double(b0[i] * ring_mod) / scale;
                    send_mg.set(i, temp_res);
                }
                CkksContext& ctx_extra = context_out->get_extra_level_context();
                Feature0DEncrypted send_ct(&ctx_extra, level + 1);
                send_ct.skip = 1;
                double encode_scale = pow(2, DEFAULT_SCALE_BIT);
                send_ct.pack_cyclic(send_mg.to_array_1d(), false, encode_scale);
                Bytes b = x_e.serialize();
                data_trans.send_bytes(b);
                Bytes send_bytes = send_ct.serialize();
                data_trans.send_bytes(send_bytes);
            } else if (type == MpcProtoType::argmax) {
                ArgMaxLayer argmax_layer(scale_ord, ring_mod, pt_range);
                argmax_layer.run(*im_0d, *im_0d);
            } else if (type == MpcProtoType::div) {
                DivLayerClient div_layer(scale_ord, ring_mod, pt_range);
                div_layer.run(*im_0d, *im_0d);
            } else if (type == MpcProtoType::reciprocal) {
                DivLayerClient div_layer(scale_ord, ring_mod, pt_range);
                div_layer.run_reciprocal(*im_0d, *im_0d);
            } else if (type == MpcProtoType::sqrt) {
                SqrtLayer sqrt_layer(scale_ord, ring_mod, pt_range);
                sqrt_layer.run(*im_0d, *im_0d);
            } else if (type == MpcProtoType::softmax) {
                SoftMaxLayerClient softmax_layer(scale_ord, ring_mod, pt_range);
                softmax_layer.run(*im_0d, *im_0d);
            } else if (type == MpcProtoType::relu) {
                Timer time_relu;
                time_relu.start();
                ReluLayerClient act(scale_ord, ring_mod, pt_range);
                act.run(*im_2d, *im_2d);
            } else if (type == MpcProtoType::simple_poly_relu) {
                int size = 0;
                io->recv_data(&size, sizeof(int));
                Array1D coeff(size);
                io->recv_data(coeff.data(), coeff.size() * sizeof(double));
                SimplePoly sp(scale_ord, ring_mod, pt_range);

                sp.init_coeff(coeff);
                sp.run(*im_2d, *im_2d);
            } else if (type == MpcProtoType::relu_0d) {
                ReluLayerClient act(scale_ord, ring_mod, pt_range);
                act.run(*im_0d, *im_0d);
            } else if (type == MpcProtoType::relu6) {
                Relu6LayerClient act(scale_ord, ring_mod, pt_range);
                act.run(*im_2d, *im_2d);
                cout << "client relu6 data=" << im_2d->data[0] << endl;
            } else if (type == MpcProtoType::max_pool) {
                im_2d = make_unique<Feature2DShare>(client_maxpool(meta_data.data[i], *im_2d, pt_range));
            } else if (type == MpcProtoType::avg_pool) {
                Duo kernel_shape;
                Duo pool_stride;
                bytes_to_va(meta_data.data[i], {"duo", "duo"}, &kernel_shape, &pool_stride);
                PoolLayerClient pool(kernel_shape, pool_stride, scale_ord, ring_mod, AVGPOOL, pt_range);
                pool.run(*im_2d, *im_2d);
            } else if (type == MpcProtoType::distance) {
                Duo array_num;
                bytes_to_va(meta_data.data[i], {"duo"}, &array_num);
                DistanceLayerClient dist(scale_ord, ring_mod, pt_range);
                Feature0DShare fs(ring_mod, scale_ord);
                dist.run(*im_0d, *im_0d, array_num);
                cout << "dist res=" << im_0d->data[0] << endl;
            } else if (type == MpcProtoType::recovery_share) {
                int data_size = im_0d->data.get_size();
                data_trans.io_in->send_data(im_0d->data.get_data(), data_size * sizeof(uint64_t));
            }
        }
    }
    cout << "@@@relu_time=" << time_relu_total << endl;
    if (im_0d == nullptr) {
        Array1DUint temp;
        return temp;
    }

    return im_0d->data.to_array_1d();
}

std::map<std::string, std::unique_ptr<CkksParameter>> init_parameters(const string& project_path) {
    std::map<std::string, std::unique_ptr<CkksParameter>> ckks_parameters;
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
    auto json_params = read_json(project_path);
    for (auto& json_param : json_params.items()) {
        const string& key = json_param.key();
        int N = json_param.value()["poly_modulus_degree"];
        int level = json_param.value()["n_mult_level"];
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
    Bytes result = ss_to_bytes(ss);
    return result;
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
    Bytes result = ss_to_bytes(ss);
    return result;
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
