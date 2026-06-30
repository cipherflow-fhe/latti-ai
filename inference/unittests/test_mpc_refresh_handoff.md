# MPC Refresh Server/Client Handoff

## Context

User asked to complete:

- `inference/unittests/test_mpc_layers.cpp::test_new_mpc_refresh_server`
- `inference/unittests/test_mpc_layers_client.cpp::test_new_mpc_refresh_client`

Goal: make the server/client refresh interaction compile and run.

## Current Edits

## Simple E2S/S2E Prototype

Added a first simple E2S/S2E path for 2D `enc_to_share` comparison:

- New proto enum: `MpcProtoType::enc_to_share_simple`.
- New proto enum: `MpcProtoType::share_to_enc_simple`.
- Client process handles `enc_to_share_simple` with `EncToShareClient::decrypt_to_share_simple`: receive encrypted masked feature, decrypt, and store each floating-point share in `Feature2DShare::data_double`.
- Client process handles `share_to_enc_simple` with `ShareToEncClient::encrypt_from_share_simple`: encrypt values from `Feature2DShare::data_double`.
- Server helper: `EncToShareServer(...).server_enc_to_share_simple(...)`, backed by `EncToShareServer::split_to_shares_simple(...)`.
- Server helper: `ShareToEncServer(...).server_share_to_enc_simple(...)`.
- Client-side S2E encryption is handled by `ShareToEncClient::encrypt_from_share_simple(...)`.
- Server-side S2E combine is handled by `ShareToEncServer::combine_with_share_simple(...)`.

Simple semantics:

```text
server samples floating-point R from `[-2^(8 + SIGMA), 2^(8 + SIGMA)]`
server sends Enc(x - R)
client decrypts to floating-point x - R
server share is floating-point R
simple S2E encrypts x - R and server adds R back
```

Implementation note: simple share values are stored as doubles inside `Feature2DShare::data_double`; this path is intended for `enc_to_share_simple -> share_to_enc_simple` roundtrip only and is not compatible with existing MPC layers that expect fixed-point ring shares in `Feature2DShare::data`.

This differs from existing complex E2S, which sends `Enc(x + r/S)` and gives server share `-r mod q`.

Added test:

```text
Feature2DEncrypted simple E2S to shares and back
```

It runs `enc_to_share_simple + share_to_enc_simple` without replacing existing complex E2S/S2E tests.

### `test_mpc_layers.cpp`

- Added missing includes for `algorithm`, `cmath`, `cstdlib`, and `random`.
- Added `get_refresh_test_port()` helper, reading `MPC_TEST_PORT` or falling back to global `port`.
- Reworked `test_new_mpc_refresh_server()` so it:
  - Initializes MPC globals as `SERVER`.
  - Receives the serialized CKKS context created by the client.
  - Uses the received context parameter `get_n()` to size test data.
  - Creates test plaintext data in the first two slots.
  - Encrypts it at level 1.
  - Masks with random `R` in `[-pow(2, 40), pow(2, 40)]` in the first two slots.
  - Encodes `R` once at input level 1, uses `sub_plain(input_ct, R_pt)`, and later reuses the same `R_pt` in `add_plain(recv_ct, R_pt)`.
  - This matches the Go example pattern `SubNew(x_ct, x1_pt)` then `AddNew(x0_ct_prime, x1_pt)`.
  - Sends masked ciphertext to client.
  - Receives refreshed ciphertext from client.
  - Adds encoded `R` back at `recv_ct.get_level()`.
  - Decrypts/decodes and checks first 32 slots.
- Added Catch test case:
  - `TEST_CASE("test_new_mpc_refresh_server", "[mpc][refresh]")`

### `test_mpc_layers_client.cpp`

- Added `cstring`.
- Replaced old port parser with parser that accepts either:
  - `./test_mpc_layers_client refresh 12309`
  - `./test_mpc_layers_client 12309`
  - `MPC_TEST_PORT=12309 ./test_mpc_layers_client refresh`
- Reworked `test_new_mpc_refresh_client(int port_in)` so it:
  - Initializes MPC globals as `CLIENT`.
  - Creates a CKKS context with `CkksParameter::create_parameter(16384)` and `support_big_complex=true`.
  - Generates rotation keys.
  - Sends the serialized context to the server.
  - Receives masked ciphertext.
  - Decrypts it.
  - Calls `context.recode_big_complex(recv_pt, 3, default_scale)`.
  - Encrypts the recoded plaintext.
  - Sends ciphertext back to server.
- Updated `main()`:
  - `refresh` first argument runs refresh client.
  - Otherwise existing `run_relu_client()` path remains the default.

## Intended Build Command

```bash
~/venv/bin/cmake --build build --target test_mpc_layers test_mpc_layers_client
```

Use the virtual environment tools under `~/venv/bin/` when compiling. The default `/usr/local/bin/cmake` in this environment may fail with:

```text
ModuleNotFoundError: No module named 'cmake'
```

## Intended Run Commands

Start server first:

```bash
MPC_TEST_PORT=12309 ./build/inference/unittests/test_mpc_layers "test_new_mpc_refresh_server"
```

Then start client:

```bash
MPC_TEST_PORT=12309 ./build/inference/unittests/test_mpc_layers_client refresh
```

Equivalent explicit-port client form:

```bash
./build/inference/unittests/test_mpc_layers_client refresh 12309
```

## Verification Status

Not compiled or executed by Codex yet. User compiled and hit a client runtime crash:

```text
panic: runtime error: invalid memory address or nil pointer dereference
main.CkksRecodeBigComplex(...)
```

Root cause found in Go SDK: `CkksRecodeBigComplex` uses `context.encoder_big`, but `DeserializeCkksContext` does not restore `support_big_complex`, so `init_ckks_context()` does not initialize `encoder_big`.

Client has been changed back to `recode_big_complex()`.
Context direction has been corrected: client creates the `support_big_complex=true` context and sends it to server; server receives and uses that context.
This avoids needing to re-enable `encoder_big` after deserialization on the client side.
 The server mask range currently follows the local C++ test value `[-pow(2, 40), pow(2, 40)]`.
Also fixed `CkksRecodeBigComplex` in Go SDK to register the returned `*ckks.Plaintext` directly with `insert_object(pt1)` instead of `insert_object(&pt1)`, which produced a `**ckks.Plaintext` handle and caused `CkksEncryptSymmetric` to panic.

Latest alignment with `lattigo/examples/ckks/enc2share2enc/main.go`:

- Client parameter changed to `N=16384` to match `PN14QP438`.
- Server now reuses one `R_pt` for both subtract and add-back, instead of re-encoding `R` at `recv_ct.get_level()`.
- Server only masks first two slots for focused checking, closer to the Go example's first-slot-only simulation.

If compilation fails, first check:

- Whether `CkksContext::serialize()` includes the secret key. The refresh client currently needs the secret key because it decrypts.
- Whether Catch filter syntax for this project accepts the quoted test name exactly as shown.
- Current C++ test flow is client-created context -> server receives context -> server sends masked ciphertext -> client recodes with `RecodeBigComplex` -> client sends refreshed ciphertext.
- For a production-safe SDK change, persist and restore `support_big_complex`, or expose a C++ method to enable/init `encoder_big` after deserialization.
