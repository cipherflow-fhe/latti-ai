#include "mpc_wrapper.h"

#include "mpc/mpc_session.h"

void init_mpc_party(int party_id, int port, const std::string& address) {
    ::mpc::init_party(party_id, port, address);
}
