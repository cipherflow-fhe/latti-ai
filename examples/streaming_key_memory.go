// Measurement-only entry point, built inside the Go SDK with a source overlay.
package main

/*
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static void *copy_eval_bytes(void *src, size_t size) {
    void *dest = malloc(size);
    if (dest != NULL) memcpy(dest, src, size);
    return dest;
}
*/
import "C"

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/cipherflow-fhe/lattigo/ckks"
)

var memoryStart = time.Now()
var memoryLog sync.Mutex

// No forced collection at key boundaries: measure the production GC behavior.
func memorySample(phase string, buffer *bytes.Buffer) {
	memoryLog.Lock()
	defer memoryLog.Unlock()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	status, err := os.ReadFile("/proc/self/status")
	if err != nil {
		panic(err)
	}
	values := map[string]any{
		"phase": phase, "seconds": time.Since(memoryStart).Seconds(),
		"heap_alloc_bytes": m.HeapAlloc, "heap_inuse_bytes": m.HeapInuse,
		"total_alloc_bytes": m.TotalAlloc, "num_gc": m.NumGC,
	}
	for _, line := range strings.Split(string(status), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 3 && (fields[0] == "VmRSS:" || fields[0] == "VmHWM:") {
			value, err := strconv.ParseUint(fields[1], 10, 64)
			if err != nil {
				panic(err)
			}
			values[strings.TrimSuffix(fields[0], ":")+"_kib"] = value
		}
	}
	if buffer != nil {
		values["buffer_cap_bytes"] = buffer.Cap()
	}
	data, err := json.Marshal(values)
	if err != nil {
		panic(err)
	}
	fmt.Println(string(data))
}

func main() {
	if len(os.Args) != 5 {
		panic("expected mode (batch-generate|batch|stream), scheme (ckks|btp), N and destination")
	}
	mode, scheme, destination := os.Args[1], os.Args[2], os.Args[4]
	n, err := strconv.ParseUint(os.Args[3], 10, 64)
	if err != nil {
		panic(err)
	}
	if mode != "batch-generate" && mode != "batch" && mode != "stream" {
		panic("invalid mode")
	}
	var parameterHandle uint64
	var params *ckks.Parameters
	var extra []int
	if scheme == "ckks" {
		parameterHandle = CreateCkksParameter(n)
		params = get_object[ckks.Parameters](parameterHandle)
	} else if scheme == "btp" {
		if n != 65536 {
			panic("BTP uses the existing default N=65536 parameter set")
		}
		parameterHandle = CreateCkksBtpParameter()
		parameters := get_object[BtpParameterSet](parameterHandle)
		params = &parameters.SchemeParam
		extra = parameters.BtpParam.RotationsForBootstrapping(*params)
	} else {
		panic("invalid scheme")
	}
	defer ReleaseHandle(parameterHandle)
	elements := evalGaloisElements(params, extra)
	sort.Slice(elements, func(i, j int) bool { return elements[i] < elements[j] })
	parameterBytes, err := params.MarshalBinary()
	if err != nil {
		panic(err)
	}
	metadata := map[string]any{
		"mode": mode, "scheme": scheme, "go_version": runtime.Version(),
		"gomaxprocs": runtime.GOMAXPROCS(0), "n": params.N(), "log_slots": params.LogSlots(),
		"q": params.Q(), "p": params.P(), "galois_elements": elements,
		"parameter_sha256": fmt.Sprintf("%x", sha256.Sum256(parameterBytes)),
		"gogc":             os.Getenv("GOGC"), "gomemlimit": os.Getenv("GOMEMLIMIT"),
	}
	if scheme == "btp" {
		btpBytes, err := get_object[BtpParameterSet](parameterHandle).BtpParam.MarshalBinary()
		if err != nil {
			panic(err)
		}
		metadata["btp_parameter_sha256"] = fmt.Sprintf("%x", sha256.Sum256(btpBytes))
	}
	data, err := json.Marshal(metadata)
	if err != nil {
		panic(err)
	}
	fmt.Println(string(data))
	memorySample("parameters", nil)
	done, stopped := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(stopped)
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				memorySample("periodic", nil)
			case <-done:
				return
			}
		}
	}()
	start := time.Now()
	var contextHandle uint64
	var outputBytes uint64
	if mode == "stream" {
		path := C.CString(destination)
		defer C.free(unsafe.Pointer(path))
		var handle C.uint64_t
		var status uint64
		if scheme == "ckks" {
			status = StreamSetupCkksContextToFile(parameterHandle, path, false, &handle)
		} else {
			status = StreamSetupCkksBtpContextToFile(parameterHandle, path, &handle)
		}
		if status != 0 {
			panic(error_message)
		}
		contextHandle = uint64(handle)
		context := get_ckks_context(contextHandle)
		if context.rlk != nil || context.gk != nil {
			panic("streaming client retained evaluation keys")
		}
		if scheme == "btp" {
			context := get_object[CkksBtpContext](contextHandle)
			if context.evk != nil || context.bootstrapper != nil {
				panic("streaming client retained a bootstrapper or evaluation bundle")
			}
		}
		file, err := os.Stat(destination)
		if err != nil {
			panic(err)
		}
		outputBytes = uint64(file.Size())
		memorySample("published", nil)
	} else {
		if scheme == "ckks" {
			contextHandle = CreateRandomCkksContext(parameterHandle, params.MaxLevel(), false)
			GenCkksContextRotationKeys(contextHandle, params.MaxLevel())
		} else {
			contextHandle = CreateRandomCkksBtpContext(parameterHandle)
			GenCkksBtpContextRotationKeys(contextHandle)
		}
		context := get_ckks_context(contextHandle)
		if context.rlk == nil || context.gk == nil || len(context.gk.Keys) != len(elements) {
			panic("batch key count mismatch")
		}
		for _, element := range elements {
			if _, ok := context.gk.Keys[element]; !ok {
				panic("batch Galois element mismatch")
			}
		}
		memorySample("generated", nil)
		if mode == "batch" {
			var publicHandle uint64
			var raw *byte
			var length C.uint64_t
			var bytesHandle uint64
			if scheme == "ckks" {
				publicHandle = MakePublicCkksContext(contextHandle, false, true, true)
				bytesHandle = SerializeCkksContextAdvanced(publicHandle, &raw, &length)
			} else {
				publicHandle = MakePublicCkksBtpContext(contextHandle)
				bytesHandle = SerializeCkksBtpContextAdvanced(publicHandle, &raw, &length)
			}
			defer ReleaseHandle(publicHandle)
			defer ReleaseHandle(bytesHandle)
			memorySample("serialized", nil)
			// Reproduce export_raw_data's native allocation and complete copy into Bytes.
			copy := C.copy_eval_bytes(unsafe.Pointer(raw), C.size_t(length))
			if copy == nil {
				panic("native evaluation context allocation failed")
			}
			defer C.free(copy)
			outputBytes = uint64(length)
			memorySample("native-copy", nil)
		}
	}
	defer ReleaseHandle(contextHandle)
	close(done)
	<-stopped
	result, err := json.Marshal(map[string]any{
		"phase": "result", "elapsed_seconds": time.Since(start).Seconds(), "output_bytes": outputBytes,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(string(result))
	// Diagnostic only, after the measured phase. Handles keep the client/legacy buffers alive.
	runtime.GC()
	memorySample("after-gc", nil)
}
