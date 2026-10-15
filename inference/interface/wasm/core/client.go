// Package core adapts native task configurations and ciphertext layouts to a
// pure Go client. File access belongs to the browser Worker.
package core

import (
	"encoding/json"
	"errors"
	"math"
	"math/bits"
	"sort"

	"github.com/cipherflow-fhe/lattigo/ckks"
	"github.com/cipherflow-fhe/lattigo/ckks/bootstrapping"
	"github.com/cipherflow-fhe/lattigo/clientcontext"
)

type FeatureParam struct {
	Dim         int             `json:"dim"`
	Channel     int             `json:"channel"`
	Shape       []int           `json:"shape"`
	Skip        json.RawMessage `json:"skip"`
	InvalidFill []int           `json:"invalid_fill"`
	PackNum     int             `json:"pack_num"`
	Level       int             `json:"level"`
	ParameterID string          `json:"ckks_parameter_id"`
	DataType    string          `json:"data_type"`
	HeadShape   []int           `json:"head_shape"`
	Heads       int             `json:"n_heads"`
	Block       int             `json:"matmul_block_size"`
	Transposed  *bool           `json:"is_transposed"`
}

type Task struct {
	UseBTP      bool                    `json:"use_btp"`
	PackStyle   string                  `json:"pack_style"`
	MatrixStyle string                  `json:"mat_pack_style"`
	BlockShape  []int                   `json:"block_shape"`
	Inputs      map[string]FeatureParam `json:"task_input_param"`
	Outputs     map[string]FeatureParam `json:"task_output_param"`
	Heads       int                     `json:"n_heads"`
	Block       int                     `json:"matmul_block_size"`
}

type Client struct {
	Task      Task
	Keys      clientcontext.Keys
	encoder   ckks.Encoder
	encryptor ckks.Encryptor
	decryptor ckks.Decryptor
}

type parameterEntry struct {
	N        int      `json:"poly_modulus_degree"`
	Q        []uint64 `json:"q"`
	P        []uint64 `json:"p"`
	LogSlots int      `json:"log_slots"`
	Name     string   `json:"param_name"`
	Levels   int      `json:"n_mult_level"`
	LogScale int      `json:"log_default_scale"`
}

func validateFixedParameters(p parameterEntry) error {
	if p.Name != "N16QP1546H192H32" || p.N != 65536 || p.Levels != 9 || p.LogScale != 40 ||
		(p.LogSlots != 0 && p.LogSlots != 15) || len(p.Q) != 0 || len(p.P) != 0 {
		return errors.New("parameters must match the N16QP1546H192H32 preset")
	}
	return nil
}

func newKeyClient(parameterBytes []byte) (*Client, error) {
	var configurations map[string]parameterEntry
	if err := json.Unmarshal(parameterBytes, &configurations); err != nil {
		return nil, err
	}
	if len(configurations) != 1 {
		return nil, errors.New("key generation requires one parameter entry")
	}
	for _, entry := range configurations {
		if err := validateFixedParameters(entry); err != nil {
			return nil, err
		}
	}
	btp := bootstrapping.N16QP1546H192H32
	params, err := ckks.NewParametersFromLiteral(btp.SchemeParams)
	if err != nil {
		return nil, err
	}
	return &Client{Keys: clientcontext.Keys{Params: params, BTP: &btp.BootstrappingParams}}, nil
}

func NewClient(taskBytes, parameterBytes []byte) (*Client, error) {
	if len(taskBytes) == 0 {
		return newKeyClient(parameterBytes)
	}
	c := new(Client)
	if err := json.Unmarshal(taskBytes, &c.Task); err != nil {
		return nil, err
	}
	if len(c.Task.Inputs) == 0 || len(c.Task.Outputs) == 0 {
		return nil, errors.New("task requires inputs and outputs")
	}
	if c.Task.PackStyle != "ordinary" && c.Task.PackStyle != "multiplexed" {
		return nil, errors.New("invalid pack_style")
	}
	// nlohmann::json orders object names; match native's first input selection.
	names := make([]string, 0, len(c.Task.Inputs))
	for name := range c.Task.Inputs {
		names = append(names, name)
	}
	sort.Strings(names)
	var configurations map[string]parameterEntry
	if err := json.Unmarshal(parameterBytes, &configurations); err != nil {
		return nil, err
	}
	entry, ok := configurations[c.Task.Inputs[names[0]].ParameterID]
	if !ok {
		return nil, errors.New("missing CKKS parameter entry")
	}
	fixedProfile := entry.Name == "N16QP1546H192H32" || (c.Task.UseBTP && entry.Name != "")
	if fixedProfile {
		if !c.Task.UseBTP {
			return nil, errors.New("fixed BTP parameter profile requires use_btp=true")
		}
		if err := validateFixedParameters(entry); err != nil {
			return nil, err
		}
	}
	var literal ckks.ParametersLiteral
	if c.Task.UseBTP {
		btp := bootstrapping.N16QP1546H192H32
		literal, c.Keys.BTP = btp.SchemeParams, &btp.BootstrappingParams
	} else if len(entry.Q) > 0 && len(entry.P) > 0 {
		if entry.N < 4096 || entry.N&(entry.N-1) != 0 {
			return nil, errors.New("invalid polynomial degree")
		}
		literal = ckks.ParametersLiteral{LogN: bits.Len(uint(entry.N)) - 1, LogSlots: bits.Len(uint(entry.N)) - 2, Q: entry.Q, P: entry.P, DefaultScale: 1 << 40}
	} else {
		switch entry.N {
		case 4096:
			literal = ckks.PN12QP109
		case 8192:
			literal = ckks.PN13QP218
		case 16384:
			literal = ckks.PN14QP438
		case 32768:
			literal = ckks.PN15QP880
		case 65536:
			literal = ckks.PN16QP1761
		default:
			return nil, errors.New("unsupported polynomial degree")
		}
	}
	params, err := ckks.NewParametersFromLiteral(literal)
	if err != nil {
		return nil, err
	}
	if !c.Task.UseBTP && entry.LogSlots > 0 {
		params, err = ckks.NewParameters(params.Parameters, entry.LogSlots, params.DefaultScale())
		if err != nil {
			return nil, err
		}
	}
	c.Keys.Params = params
	for _, features := range []map[string]FeatureParam{c.Task.Inputs, c.Task.Outputs} {
		for name, p := range features {
			if fixedProfile {
				referenced, ok := configurations[p.ParameterID]
				if !ok || validateFixedParameters(referenced) != nil {
					return nil, errors.New("task parameter reference does not match the fixed profile")
				}
			}
			if p.Channel == 0 {
				p.Channel = 1
			}
			if p.Heads == 0 {
				p.Heads = c.Task.Heads
			}
			if p.Block == 0 {
				p.Block = c.Task.Block
			}
			if p.Dim < 0 || p.Dim > 2 || p.Channel < 1 || p.Level < 0 || p.Level > params.MaxLevel() {
				return nil, errors.New("invalid feature dimensions or level")
			}
			if p.Dim > 0 && (len(p.Shape) != p.Dim || product(p.Shape) <= 0) {
				return nil, errors.New("invalid feature shape")
			}
			features[name] = p
		}
	}
	return c, nil
}

func (c *Client) initCrypto() {
	c.encoder = ckks.NewEncoder(c.Keys.Params)
	c.encryptor = ckks.NewEncryptor(c.Keys.Params, c.Keys.PK)
	c.decryptor = ckks.NewDecryptor(c.Keys.Params, c.Keys.SK)
}

func (c *Client) BeginSetup() (*clientcontext.EvaluationStream, error) {
	kgen := ckks.NewKeyGenerator(c.Keys.Params)
	c.Keys.SK, c.Keys.PK = kgen.GenKeyPair()
	c.initCrypto()
	return clientcontext.NewEvaluationStream(c.Keys)
}

func (c *Client) Full() ([]byte, error) { return clientcontext.MarshalFull(c.Keys) }

func (c *Client) Restore(data []byte) error {
	keys, err := clientcontext.UnmarshalFull(data, c.Keys.Params, c.Keys.BTP)
	if err != nil {
		return err
	}
	c.Keys = keys
	c.initCrypto()
	return nil
}

func (c *Client) Encrypt(name string, values []float64) ([]byte, error) {
	if len(c.Task.Inputs) == 0 {
		return nil, errors.New("task configuration is required for encryption")
	}
	if c.encryptor == nil {
		return nil, errors.New("setup or restore required")
	}
	p, ok := c.Task.Inputs[name]
	if !ok {
		return nil, errors.New("unknown input name")
	}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, errors.New("non-finite input")
		}
	}
	f, slots, err := c.pack(p, values)
	if err != nil {
		return nil, err
	}
	scale := math.Exp2(math.Round(math.Log2(float64(c.Keys.Params.Q()[1]))))
	for _, vector := range slots {
		pt := c.encoder.EncodeNew(vector, p.Level, scale, c.Keys.Params.LogSlots())
		f.ciphertexts = append(f.ciphertexts, c.encryptor.EncryptNew(pt))
	}
	return f.marshal(c.Keys.Params), nil
}

func (c *Client) Decrypt(name string, raw []byte) (values []float64, err error) {
	if len(c.Task.Outputs) == 0 {
		return nil, errors.New("task configuration is required for decryption")
	}
	if c.decryptor == nil {
		return nil, errors.New("setup or restore required")
	}
	p, ok := c.Task.Outputs[name]
	if !ok {
		return nil, errors.New("unknown output name")
	}
	f, err := unmarshalFeature(raw, c.Keys.Params, p.DataType == "feature_mat")
	if err != nil {
		return nil, err
	}
	if f.dim != p.Dim {
		return nil, errors.New("output dimension mismatch")
	}
	slots := make([][]float64, len(f.ciphertexts))
	for i, ct := range f.ciphertexts {
		decoded := c.encoder.Decode(c.decryptor.DecryptNew(ct), c.Keys.Params.LogSlots())
		slots[i] = make([]float64, len(decoded))
		for j, value := range decoded {
			slots[i][j] = real(value)
		}
	}
	return c.unpack(p, f, slots)
}

func product(v []int) int {
	n := 1
	for _, x := range v {
		if x <= 0 || n > (1<<26)/x {
			return 0
		}
		n *= x
	}
	return n
}
func ceil(x, y int) int { return (x + y - 1) / y }
func pow2(x int) int    { return 1 << bits.Len(uint(x-1)) }
func (p FeatureParam) skip() []int {
	v := []int{1, 1}
	if len(p.Skip) == 0 {
		return v
	}
	var scalar int
	if json.Unmarshal(p.Skip, &scalar) == nil {
		v[0] = scalar
		return v
	}
	var array []int
	if json.Unmarshal(p.Skip, &array) == nil {
		copy(v, array)
	}
	return v
}
