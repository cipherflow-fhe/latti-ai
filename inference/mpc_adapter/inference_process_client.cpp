#include "inference_process_client.h"
#include "mpc_adapter/enc_share_conversion.h"
#include "mpc/fhe_mpc.h"
#include "mpc_array_bridge.h"
#include "mpc/mpc_session.h"
#include "mpc/mpc_task_meta_data.h"
#include "util.h"

using namespace std;

namespace {

Feature2DShare client_maxpool(const Bytes& meta_data_bytes, const Feature2DShare& x, double pt_range) {
    Duo kernel_shape;
    Duo pool_stride;
    bytes_to_va(meta_data_bytes, {"duo", "duo"}, &kernel_shape, &pool_stride);

    PoolLayerClient pool(kernel_shape, pool_stride, DEFAULT_SCALE_BIT, RING_MOD, MAXPOOL, pt_range);
    Feature2DShare y(x.ring_mod, x.scale_ord);
    int num_matrix = x.data.get_size() / x.shape[0] / x.shape[1];
    assign_share_data_from_mpc_array(y.data, pool.run(x.data.to_array_1d(), x.shape, num_matrix));
    y.shape = pool.output_shape(x.shape);

    return y;
}

}  // namespace

InferenceMpcClient::InferenceMpcClient(map<string, unique_ptr<CkksContext>>& ckks_contexts)
    : ckks_contexts_(ckks_contexts) {}

mpc::Array1DUint InferenceMpcClient::run() {
    DataTransmission data_trans = ::mpc::data_transmission();

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
        EncToShareClient enc_to_share_client(ckks_contexts_, context_in, context_out, scale_ord_, ring_mod_);

        for (int i = 0; i < n_proto; i++) {
            MpcProtoType type = meta_data.types[i];
            if (type == MpcProtoType::end) {
                if (im_0d == nullptr) {
                    mpc::Array1DUint temp;
                    return temp;
                }
                return im_0d->data.to_array_1d();
            } else if (type == MpcProtoType::enc_to_share || type == MpcProtoType::enc_to_share_simple) {
                if (type == MpcProtoType::enc_to_share_simple) {
                    im_2d = make_unique<Feature2DShare>(
                        enc_to_share_client.client_enc_to_share_simple(meta_data.data[i]));
                } else {
                    im_2d = make_unique<Feature2DShare>(enc_to_share_client.client_enc_to_share(meta_data.data[i]));
                }

            } else if (type == MpcProtoType::enc_to_share_for_multi_channel_pack) {
                im_2d = make_unique<Feature2DShare>(
                    enc_to_share_client.client_enc_to_share_for_multi_channel_pack(meta_data.data[i]));
            } else if (type == MpcProtoType::enc_to_share_for_multi_channel_pack_simple) {
                im_2d = make_unique<Feature2DShare>(
                    enc_to_share_client.client_enc_to_share_for_multi_channel_pack_simple(meta_data.data[i]));
            } else if (type == MpcProtoType::enc_to_share_0d) {
                im_0d = make_unique<Feature0DShare>(enc_to_share_client.client_enc_to_share_0d(meta_data.data[i]));
            } else if (type == MpcProtoType::share_to_enc) {
                ShareToEncClient share_to_enc_client(*context_out, scale_ord_, ring_mod_, pt_range_);
                share_to_enc_client.client_share_to_enc(*im_2d, meta_data.data[i]);
            } else if (type == MpcProtoType::share_to_enc_simple) {
                ShareToEncClient share_to_enc_client(*context_out, scale_ord_, ring_mod_, pt_range_);
                share_to_enc_client.client_share_to_enc_simple(*im_2d, meta_data.data[i]);
            } else if (type == MpcProtoType::share_to_enc_for_multi_channel_pack) {
                ShareToEncClient share_to_enc_client(*context_out, scale_ord_, ring_mod_, pt_range_);
                share_to_enc_client.client_share_to_enc_for_multi_channel_pack(*im_2d, meta_data.data[i]);
            } else if (type == MpcProtoType::share_to_enc_for_multi_channel_pack_simple) {
                ShareToEncClient share_to_enc_client(*context_out, scale_ord_, ring_mod_, pt_range_);
                share_to_enc_client.client_share_to_enc_for_multi_channel_pack_simple(*im_2d, meta_data.data[i]);
            } else if (type == MpcProtoType::share_2d_to_0d) {
                im_0d = make_unique<Feature0DShare>(ring_mod_, scale_ord_);
                im_0d->data = std::move(im_2d->data);
            } else if (type == MpcProtoType::share_to_enc_0d) {
                ShareToEncClient share_to_enc_client(*context_out, scale_ord_, ring_mod_, pt_range_);
                share_to_enc_client.client_share_to_enc_0d(*im_0d, meta_data.data[i]);
            } else if (type == MpcProtoType::argmax) {
                ArgMaxLayer argmax_layer(scale_ord_, ring_mod_, pt_range_);
                im_0d->data = decltype(im_0d->data)::from_array_1d(argmax_layer.run(im_0d->data.to_array_1d()));
            } else if (type == MpcProtoType::div) {
                DivLayerClient div_layer(scale_ord_, ring_mod_, pt_range_);
                im_0d->data = decltype(im_0d->data)::from_array_1d(div_layer.run(im_0d->data.to_array_1d()));
            } else if (type == MpcProtoType::reciprocal) {
                DivLayerClient div_layer(scale_ord_, ring_mod_, pt_range_);
                im_0d->data =
                    decltype(im_0d->data)::from_array_1d(div_layer.run_reciprocal(im_0d->data.to_array_1d()));
            } else if (type == MpcProtoType::sqrt) {
                SqrtLayer sqrt_layer(scale_ord_, ring_mod_, pt_range_);
                im_0d->data = decltype(im_0d->data)::from_array_1d(sqrt_layer.run(im_0d->data.to_array_1d()));
            } else if (type == MpcProtoType::softmax) {
                SoftMaxLayerClient softmax_layer(scale_ord_, ring_mod_, pt_range_);
                im_0d->data = decltype(im_0d->data)::from_array_1d(softmax_layer.run(im_0d->data.to_array_1d()));
            } else if (type == MpcProtoType::relu) {
                ReluLayerClient act(scale_ord_, ring_mod_, pt_range_);
                assign_share_data_from_mpc_array(im_2d->data, act.run(share_data_to_mpc_array(im_2d->data)));
            } else if (type == MpcProtoType::simple_poly_relu) {
                int size = 0;
                data_trans.recv_data(&size, sizeof(int));
                Array1D coeff(size);
                data_trans.recv_data(coeff.data(), coeff.size() * sizeof(double));
                SimplePoly sp(scale_ord_, ring_mod_, pt_range_);

                sp.init_coeff(coeff);
                assign_share_data_from_mpc_array(im_2d->data, sp.run(share_data_to_mpc_array(im_2d->data)));
            } else if (type == MpcProtoType::relu_0d) {
                ReluLayerClient act(scale_ord_, ring_mod_, pt_range_);
                assign_share_data_from_mpc_array(im_0d->data, act.run(share_data_to_mpc_array(im_0d->data)));
            } else if (type == MpcProtoType::relu6) {
                Relu6LayerClient act(scale_ord_, ring_mod_, pt_range_);
                assign_share_data_from_mpc_array(im_2d->data, act.run(share_data_to_mpc_array(im_2d->data)));
                cout << "client relu6 data=" << im_2d->data[0] << endl;
            } else if (type == MpcProtoType::max_pool) {
                im_2d = make_unique<Feature2DShare>(client_maxpool(meta_data.data[i], *im_2d, pt_range_));
            } else if (type == MpcProtoType::avg_pool) {
                Duo kernel_shape;
                Duo pool_stride;
                bytes_to_va(meta_data.data[i], {"duo", "duo"}, &kernel_shape, &pool_stride);
                PoolLayerClient pool(kernel_shape, pool_stride, scale_ord_, ring_mod_, AVGPOOL, pt_range_);
                int num_matrix = im_2d->data.get_size() / im_2d->shape[0] / im_2d->shape[1];
                auto input_shape = im_2d->shape;
                assign_share_data_from_mpc_array(
                    im_2d->data, pool.run(im_2d->data.to_array_1d(), input_shape, num_matrix));
                im_2d->shape = pool.output_shape(input_shape);
            } else if (type == MpcProtoType::distance) {
                Duo array_num;
                bytes_to_va(meta_data.data[i], {"duo"}, &array_num);
                DistanceLayerClient dist(scale_ord_, ring_mod_, pt_range_);
                Feature0DShare fs(ring_mod_, scale_ord_);
                assign_share_data_from_mpc_array(
                    im_0d->data, dist.run(share_data_to_mpc_array(im_0d->data), array_num));
                cout << "dist res=" << im_0d->data[0] << endl;
            } else if (type == MpcProtoType::recovery_share) {
                int data_size = im_0d->data.get_size();
                data_trans.send_data(im_0d->data.get_data(), data_size * sizeof(uint64_t));
            }
        }
    }
    if (im_0d == nullptr) {
        mpc::Array1DUint temp;
        return temp;
    }

    return im_0d->data.to_array_1d();
}
