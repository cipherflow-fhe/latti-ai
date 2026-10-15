//go:build js && wasm

package main

import (
	"runtime"
	"syscall/js"

	"github.com/cipherflow-fhe/latti-ai/client-wasm/core"
	"github.com/cipherflow-fhe/lattigo/clientcontext"
)

const chunkSize = 4 << 20

var client *core.Client
var stream *clientcontext.EvaluationStream
var payload []byte
var payloadOffset, payloadPeak int

func byteArray(data []byte) js.Value {
	array := js.Global().Get("Uint8Array").New(len(data))
	js.CopyBytesToJS(array, data)
	return array
}

func bytesFrom(value js.Value) []byte {
	data := make([]byte, value.Get("byteLength").Int())
	js.CopyBytesToGo(data, value)
	return data
}

func call(_ js.Value, args []js.Value) (result any) {
	defer func() {
		if recover() != nil {
			result = map[string]any{"error": "WASM client operation failed"}
		}
	}()
	var err error
	var data any
	switch args[0].String() {
	case "init", "init-keys":
		stream, payload = nil, nil
		payloadOffset, payloadPeak = 0, 0
		if args[0].String() == "init-keys" {
			client, err = core.NewClient(nil, []byte(args[1].String()))
		} else {
			client, err = core.NewClient([]byte(args[1].String()), []byte(args[2].String()))
		}
	case "begin":
		stream, err = client.BeginSetup()
	case "next":
		if stream == nil {
			return map[string]any{"error": "setup not started"}
		}
		if payloadOffset == len(payload) {
			payload = nil
			payload, err = stream.Next()
			payloadOffset = 0
			if len(payload) > payloadPeak {
				payloadPeak = len(payload)
			}
		}
		if err == nil {
			if payload == nil {
				stream = nil
				runtime.GC() // release the final payload before full-context export
				data = map[string]any{"done": true}
			} else {
				end := min(payloadOffset+chunkSize, len(payload))
				data = map[string]any{"done": false, "chunk": byteArray(payload[payloadOffset:end])}
				payloadOffset = end
			}
		}
	case "full":
		var raw []byte
		raw, err = client.Full()
		if err == nil {
			data = byteArray(raw)
		}
	case "restore":
		err = client.Restore(bytesFrom(args[1]))
	case "encrypt":
		array := args[2]
		values := make([]float64, array.Length())
		for i := range values {
			values[i] = array.Index(i).Float()
		}
		var raw []byte
		raw, err = client.Encrypt(args[1].String(), values)
		if err == nil {
			data = byteArray(raw)
		}
	case "decrypt":
		var values []float64
		values, err = client.Decrypt(args[1].String(), bytesFrom(args[2]))
		if err == nil {
			array := js.Global().Get("Float64Array").New(len(values))
			for i, value := range values {
				array.SetIndex(i, value)
			}
			data = array
		}
	case "close":
		client, stream, payload = nil, nil, nil
		runtime.GC()
	default:
		return map[string]any{"error": "unknown WASM operation"}
	}
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	return map[string]any{"data": data, "heapBytes": float64(memory.HeapAlloc), "payloadBytes": payloadPeak}
}

func main() {
	js.Global().Set("lattiCore", js.FuncOf(call))
	select {}
}
