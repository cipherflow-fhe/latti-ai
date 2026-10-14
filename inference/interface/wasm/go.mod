module github.com/cipherflow-fhe/latti-ai/client-wasm

go 1.24.0

toolchain go1.24.11

replace github.com/cipherflow-fhe/lattigo => ../../lattisense/fhe_ops_lib/lattigo

require github.com/cipherflow-fhe/lattigo v0.0.0-00010101000000-000000000000

require (
	golang.org/x/crypto v0.47.0 // indirect
	golang.org/x/sys v0.40.0 // indirect
)
