package core

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"strconv"
	"testing"

	"github.com/cipherflow-fhe/lattigo/schemes/ckks"
)

func TestProvisionStreamsSignatureLevelsScalesAndNativeContainers(t *testing.T) {
	client, err := NewKeyClient(testParameters(t, "p", 1), false)
	if err != nil {
		t.Fatal(err)
	}
	if err = client.GenerateKeys(); err != nil {
		t.Fatal(err)
	}
	client.Task.DeploymentMode = "server_provisioned_runner"
	slots := client.Keys.Params.MaxSlots()
	plan := []map[string]any{
		{"id": "weight", "type": "ct", "size": []int{2}, "count": 2, "slots": slots, "level": 1, "scale": float64(client.Keys.Params.Q()[1])},
		{"id": "bias", "type": "ct", "size": []int{1}, "count": 1, "slots": slots, "level": 0, "scale": client.Keys.Params.DefaultScale().Float64()},
	}
	rawPlan, _ := json.Marshal(plan)
	stream, err := client.StartProvision(rawPlan)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Cancel()
	outputs := map[string]*bytes.Buffer{}
	event := streamEvent(t, stream, nil)
	reads := 0
	for event.Type != "done" {
		switch event.Type {
		case "need-input":
			if event.Offset != "0" || event.MaxBytes != slots*8 {
				t.Fatal("materializer did not request one full vector", event)
			}
			data := make([]byte, slots*8)
			for i := range slots {
				binary.LittleEndian.PutUint64(data[i*8:], math.Float64bits(float64(reads+1)/8))
			}
			reads++
			event, err = stream.NextInput(nil, data, true)
			if err != nil {
				t.Fatal(err)
			}
			continue
		case "begin-output":
			outputs[event.OutputID] = new(bytes.Buffer)
		case "write-chunk":
			if len(event.Data) > 4*1024*1024 {
				t.Fatal("unbounded output")
			}
			outputs[event.OutputID].Write(event.Data)
		case "end-output":
		default:
			t.Fatal(event.Type, event.Err)
		}
		event = streamEvent(t, stream, nil)
	}
	if reads != 3 {
		t.Fatal("wrong parameter flat count")
	}
	var manifest struct {
		Format    string `json:"format"`
		Arguments map[string]struct {
			Elements []struct{ Offset, Length uint64 } `json:"elements"`
		} `json:"arguments"`
	}
	if err := json.Unmarshal(outputs["parameter-manifest"].Bytes(), &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Format != "latti-ai.encrypted-parameter-store.v2" {
		t.Fatal(manifest.Format)
	}
	valueIndex := 0
	for index, argument := range plan {
		reader := bytes.NewReader(outputs["parameter-"+strconv.Itoa(index)].Bytes())
		var count uint64
		if err := binary.Read(reader, binary.LittleEndian, &count); err != nil {
			t.Fatal(err)
		}
		if count != uint64(argument["count"].(int)) {
			t.Fatal("invalid container count")
		}
		for flat := uint64(0); flat < count; flat++ {
			var length uint64
			if err := binary.Read(reader, binary.LittleEndian, &length); err != nil {
				t.Fatal(err)
			}
			span := manifest.Arguments[argument["id"].(string)].Elements[flat]
			position, _ := reader.Seek(0, 1)
			if span.Offset != uint64(position) || span.Length != length {
				t.Fatal("invalid ciphertext index")
			}
			data := make([]byte, length)
			if _, err := reader.Read(data); err != nil {
				t.Fatal(err)
			}
			ciphertext := ckks.NewCiphertext(client.Keys.Params, 1, argument["level"].(int))
			if err := ciphertext.UnmarshalBinary(data); err != nil {
				t.Fatal(err)
			}
			if ciphertext.Level() != argument["level"].(int) || ciphertext.Scale.Float64() != argument["scale"].(float64) {
				t.Fatal("parameter level/scale changed")
			}
			decoded := make([]float64, slots)
			if err := client.encoder.Decode(client.decryptor.DecryptNew(ciphertext), decoded); err != nil {
				t.Fatal(err)
			}
			valueIndex++
			for _, value := range decoded {
				if math.Abs(value-float64(valueIndex)/8) > 1e-6 {
					t.Fatal("parameter value differs", value)
				}
			}
		}
		if reader.Len() != 0 {
			t.Fatal("trailing parameter bytes")
		}
	}
}

func TestProvisionRejectsInvalidPlansAndMissingElements(t *testing.T) {
	client, err := NewKeyClient(testParameters(t, "p", 1), false)
	if err != nil {
		t.Fatal(err)
	}
	if err = client.GenerateKeys(); err != nil {
		t.Fatal(err)
	}
	client.Task.DeploymentMode = "server_provisioned_runner"
	valid := map[string]any{"id": "weight", "type": "ct", "size": []int{1}, "count": 1,
		"slots": client.Keys.Params.MaxSlots(), "level": 1, "scale": client.Keys.Params.DefaultScale().Float64()}
	for _, field := range []string{"count", "slots", "level", "scale", "type", "duplicate"} {
		plan := map[string]any{}
		for name, value := range valid {
			plan[name] = value
		}
		switch field {
		case "count":
			plan[field] = 2
		case "slots":
			plan[field] = 1
		case "level":
			plan[field] = -1
		case "scale":
			plan[field] = 0
		case "type":
			plan[field] = "pt"
		}
		arguments := []map[string]any{plan}
		if field == "duplicate" {
			arguments = append(arguments, plan)
		}
		raw, _ := json.Marshal(arguments)
		if stream, failure := client.StartProvision(raw); failure == nil {
			stream.Cancel()
			t.Fatal("invalid plan accepted", field)
		}
	}
	raw, _ := json.Marshal([]map[string]any{valid})
	stream, err := client.StartProvision(raw)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Cancel()
	event := streamEvent(t, stream, nil)
	for event.Type != "need-input" {
		if event.Type == "error" || event.Type == "done" {
			t.Fatal("missing parameter request", event)
		}
		event = streamEvent(t, stream, nil)
	}
	event, err = stream.NextInput(nil, make([]byte, event.MaxBytes-8), true)
	if err != nil {
		t.Fatal(err)
	}
	if event.Type != "error" {
		t.Fatal("short parameter vector accepted", event)
	}
}
