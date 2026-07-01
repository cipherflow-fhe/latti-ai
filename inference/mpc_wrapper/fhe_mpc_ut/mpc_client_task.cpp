#include "mpc_wrapper/inference_process_client.h"
#include "mpc_wrapper/mpc_data_transmission.h"
#include "mpc_wrapper/mpc_wrapper.h"

#include <iostream>

using namespace std;

int main() {
    init_mpc_party(MPC_CLIENT, 12309);
    int N = 8192;
    CkksParameter param = CkksParameter::create_parameter(N);
    CkksContext context = CkksContext::create_random_context(param);

    context.gen_rotation_keys();

    MpcDataTransmission::current().send_public_context(context);
    cout << "send_public_context ok" << endl;
    map<string, unique_ptr<CkksContext>> ckks_contexts;
    ckks_contexts["param0"] = make_unique<CkksContext>(move(context));

    InferenceMpcClient(ckks_contexts).run();
}
