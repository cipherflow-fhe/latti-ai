//go:build js && wasm

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"strconv"
	"syscall/js"

	"github.com/cipherflow-fhe/latti-ai/client-wasm/core"
)

var client *core.Client
var stream *core.OutputStream
var operationID uint64
var runtimeVersion, buildID string

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
		if stream != nil {
			return map[string]any{"error": "operation already active"}
		}
		if args[0].String() == "init-keys" {
			client, err = core.NewKeyClient([]byte(args[1].String()), args[2].Bool())
		} else {
			client, err = core.NewClient([]byte(args[1].String()), []byte(args[2].String()))
		}
	case "profile":
		if client == nil {
			err = errors.New("client not initialized")
		} else {
			var profile map[string]any
			profile, err = client.Profile()
			if err == nil {
				var raw []byte
				raw, err = json.Marshal(profile)
				if err == nil {
					data = js.Global().Get("JSON").Call("parse", string(raw))
				}
			}
		}
	case "info":
		data = map[string]any{"runtimeVersion": runtimeVersion, "protocolVersion": 2, "build": buildID}
	case "start":
		if stream != nil {
			return map[string]any{"error": "operation already active"}
		}
		if client == nil {
			return map[string]any{"error": "client not initialized"}
		}
		switch args[1].String() {
		case "setup":
			stream, err = client.StartSetup()
		case "encrypt", "decrypt", "restore":
			var size uint64
			size, err = strconv.ParseUint(args[3].String(), 10, 64)
			if err == nil {
				if args[1].String() == "restore" {
					stream = client.StartRestore(size)
				} else if args[1].String() == "encrypt" {
					stream, err = client.StartEncrypt(args[2].String(), size)
				} else {
					stream, err = client.StartDecrypt(args[2].String(), size)
				}
			}
		default:
			err = fmt.Errorf("unknown streaming operation")
		}
		if err == nil {
			operationID++
			data = strconv.FormatUint(operationID, 10)
		}
	case "next", "cancel":
		if stream == nil || len(args) < 2 || args[1].String() != strconv.FormatUint(operationID, 10) {
			return map[string]any{"error": "unknown operation ID"}
		}
		if args[0].String() == "cancel" {
			stream.Cancel()
		}
		var ack error
		if len(args) > 2 && args[2].String() != "" {
			ack = errors.New(args[2].String())
		}
		var event core.OutputEvent
		if len(args) > 3 && args[3].Type() != js.TypeUndefined && args[3].Type() != js.TypeNull {
			if args[3].Get("byteLength").Int() > 4*1024*1024 {
				return map[string]any{"error": "input chunk too large"}
			}
			event, err = stream.NextInput(ack, bytesFrom(args[3]), args[4].Bool())
		} else {
			event, err = stream.Next(ack)
		}
		if err == nil {
			data = map[string]any{"type": event.Type, "outputId": event.OutputID, "offset": event.Offset, "bytes": event.Bytes, "sourceId": event.SourceID, "maxBytes": event.MaxBytes, "role": event.Role, "format": event.Format}
			if event.Data != nil {
				data.(map[string]any)["chunk"] = byteArray(event.Data)
			}
			if event.Err != nil {
				data.(map[string]any)["message"] = event.Err.Error()
			}
			if event.Type == "done" || event.Type == "error" || event.Type == "cancelled" {
				stream = nil
			}
		}
	case "bind-task":
		if client == nil || stream != nil {
			err = errors.New("client unavailable or busy")
		} else {
			err = client.BindTask([]byte(args[1].String()), []byte(args[2].String()))
		}
	case "close":
		if stream != nil {
			stream.Cancel()
			_, err = stream.Next(nil)
			stream = nil
		}
		client = nil
		runtime.GC()
	default:
		return map[string]any{"error": "unknown WASM operation"}
	}
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	return map[string]any{"data": data, "heapBytes": float64(memory.HeapAlloc), "payloadBytes": 0}
}

func main() {
	js.Global().Set("lattiCore", js.FuncOf(call))
	select {}
}
