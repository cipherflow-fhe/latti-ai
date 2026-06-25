#include "inference_process_client.h"
#include "mpc_data_transmission.h"

using namespace std;

int party = CLIENT;
int port = 12309;
string address = "127.0.0.1";
int num_threads = 1;

int main() {
    party = CLIENT;
    port = 12309;
    address = "127.0.0.1";
    num_threads = 1;
    bitlength = RING_MOD_BIT;
    StartComputation();
    int N = 8192;
    CkksParameter param = CkksParameter::create_parameter(N);
    CkksContext context = CkksContext::create_random_context(param);

    context.gen_rotation_keys();

    DataTransmission dt(io);
    send_public_context(dt, context);
    cout << "send_public_context ok" << endl;
    map<string, unique_ptr<CkksContext>> ckks_contexts;
    ckks_contexts["param0"] = make_unique<CkksContext>(move(context));

    process(&ckks_contexts);
}
