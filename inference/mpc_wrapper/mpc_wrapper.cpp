#include "mpc_wrapper.h"

#include "mpc/mpc_session.h"
#include "mpc/seconnds_relu_abi.h"

void init_mpc_party(int party_id, int port, const std::string& address) {
    ::mpc::init_party(party_id, port, address);
    if constexpr (::mpc::USE_SECONNDS_RELU) {
        ::mpc::init_mpc_seconnds_relu(party_id);
    }
}
