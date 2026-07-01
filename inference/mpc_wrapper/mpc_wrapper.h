#pragma once

#include <string>

constexpr int MPC_SERVER = 1;
constexpr int MPC_CLIENT = 2;

void init_mpc_party(int party_id, int port, const std::string& address = "127.0.0.1");
