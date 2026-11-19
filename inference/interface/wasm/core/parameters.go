package core

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"

	"github.com/cipherflow-fhe/lattigo/clientcontext"
	"github.com/cipherflow-fhe/lattigo/core/rlwe"
	"github.com/cipherflow-fhe/lattigo/schemes/ckks"
)

func StartSHA256(size uint64) *OutputStream {
	return startOutput(func(stream *OutputStream) error {
		source := &inputSource{stream: stream, id: "bytes", size: size}
		hash := sha256.New()
		if _, err := io.CopyBuffer(hash, source, make([]byte, 64*1024)); err != nil {
			return err
		}
		return stream.output("sha256", func(sink clientcontext.Sink) (uint64, error) {
			n, err := fmt.Fprintf(sink, "%x", hash.Sum(nil))
			return uint64(n), err
		})
	})
}

type parameterArgument struct {
	ID    string  `json:"id"`
	Size  []int   `json:"size"`
	Count int     `json:"count"`
	Slots int     `json:"slots"`
	Level int     `json:"level"`
	Scale float64 `json:"scale"`
}

// StartProvision consumes one already materialized RingT slot vector at a time.
// Scale and use-site levels come from the signature, independently of Feature
// input packing. The container's raw ciphertext bytes are native Lattigo v6.
func (c *Client) StartProvision(plan []byte) (*OutputStream, error) {
	if c.Task.DeploymentMode != "server_provisioned_runner" || c.encryptor == nil || c.decryptor == nil {
		return nil, errors.New("parameter mode and setup/restore required")
	}
	var arguments []parameterArgument
	var metadata []map[string]json.RawMessage
	if err := json.Unmarshal(plan, &arguments); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(plan, &metadata); err != nil {
		return nil, err
	}
	if len(arguments) == 0 || len(arguments) > 100000 {
		return nil, errors.New("invalid parameter argument count")
	}
	names := map[string]bool{}
	total := 0
	for index, arg := range arguments {
		if arg.ID == "" || names[arg.ID] || arg.Count < 1 || product(arg.Size) != arg.Count ||
			arg.Slots != c.Keys.Params.MaxSlots() || arg.Slots*8 > clientcontext.MaxChunkBytes ||
			arg.Level < 0 || arg.Level > c.Keys.Params.MaxLevel() ||
			math.IsNaN(arg.Scale) || math.IsInf(arg.Scale, 0) || arg.Scale <= 0 {
			return nil, errors.New("invalid parameter signature geometry/level/scale")
		}
		total += arg.Count
		if total > 16384 {
			return nil, errors.New("parameter index exceeds the validated browser profile")
		}
		if string(metadata[index]["type"]) != `"ct"` {
			return nil, errors.New("parameter signature must declare a ciphertext")
		}
		names[arg.ID] = true
	}
	return startOutput(func(stream *OutputStream) error {
		manifest := map[string]any{"format": "latti-ai.encrypted-parameter-store.v2"}
		entries := map[string]any{}
		manifest["arguments"] = entries
		for index, arg := range arguments {
			elements := make([]map[string]uint64, 0, arg.Count)
			maxError, maxCiphertextBytes := 0.0, uint64(0)
			outputID := "parameter-" + strconv.Itoa(index)
			err := stream.output(outputID, func(sink clientcontext.Sink) (uint64, error) {
				out := bufio.NewWriterSize(sink, 64*1024)
				if err := binary.Write(out, binary.LittleEndian, uint64(arg.Count)); err != nil {
					return 0, err
				}
				offset := uint64(8)
				for flat := 0; flat < arg.Count; flat++ {
					if err := stream.ctx.Err(); err != nil {
						return 0, err
					}
					source := &inputSource{stream: stream, id: fmt.Sprintf("parameter:%d:%d", index, flat), size: uint64(arg.Slots * 8)}
					raw, err := source.readAt(0, arg.Slots*8)
					if err != nil {
						return 0, err
					}
					vector := make([]float64, arg.Slots)
					for i := range vector {
						value := math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
						if math.IsNaN(value) || math.IsInf(value, 0) {
							return 0, errors.New("nonfinite parameter value")
						}
						vector[i] = value
					}
					pt := ckks.NewPlaintext(c.Keys.Params, arg.Level)
					pt.Scale = rlwe.NewScale(arg.Scale)
					if err := c.encoder.Encode(vector, pt); err != nil {
						return 0, err
					}
					ciphertext, err := c.encryptor.EncryptNew(pt)
					if err != nil {
						return 0, err
					}
					decoded := make([]float64, arg.Slots)
					if err := c.encoder.Decode(c.decryptor.DecryptNew(ciphertext), decoded); err != nil {
						return 0, err
					}
					for i, value := range decoded {
						error := math.Abs(value - vector[i])
						if math.IsNaN(error) || math.IsInf(error, 0) || error >= 1e-5 {
							return 0, errors.New("parameter encoding roundtrip exceeds 1e-5")
						}
						maxError = max(maxError, error)
					}
					length := uint64(ciphertext.BinarySize())
					maxCiphertextBytes = max(maxCiphertextBytes, length)
					if err := binary.Write(out, binary.LittleEndian, length); err != nil {
						return 0, err
					}
					written, err := ciphertext.WriteTo(out)
					if err != nil || uint64(written) != length {
						return 0, errors.New("parameter ciphertext serialization failed")
					}
					elements = append(elements, map[string]uint64{"offset": offset + 8, "length": length})
					offset += 8 + length
					// Finish disk acknowledgement before producing another ciphertext.
					if err := out.Flush(); err != nil {
						return 0, err
					}
				}
				return offset, out.Flush()
			})
			if err != nil {
				return err
			}
			entry := map[string]any{}
			for key, value := range metadata[index] {
				entry[key] = value
			}
			entry["file"], entry["elements"] = outputID+".bin", elements
			entry["verified_count"], entry["encoding_max_abs_error"] = arg.Count, maxError
			entry["max_ciphertext_bytes"] = maxCiphertextBytes
			entries[arg.ID] = entry
		}
		return stream.output("parameter-manifest", func(sink clientcontext.Sink) (uint64, error) {
			// uint64 offsets are serialized here and never parsed/stringified in JS.
			if err := json.NewEncoder(sink).Encode(manifest); err != nil {
				return 0, err
			}
			return sink.(*outputSink).pos, nil
		})
	}), nil
}
