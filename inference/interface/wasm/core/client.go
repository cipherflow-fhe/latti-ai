package core

import (
	"encoding/json"
	"errors"
	"math/bits"
	"sort"

	"github.com/cipherflow-fhe/lattigo/clientcontext"
	"github.com/cipherflow-fhe/lattigo/core/rlwe"
	"github.com/cipherflow-fhe/lattigo/schemes/ckks"
)

type FeatureParam struct {
	Dim              int             `json:"dim"`
	Channel          int             `json:"channel"`
	Shape            []int           `json:"shape"`
	Skip             json.RawMessage `json:"skip"`
	InvalidFill      []int           `json:"invalid_fill"`
	PackNum          int             `json:"pack_num"`
	Level            int             `json:"level"`
	ParameterID      string          `json:"ckks_parameter_id"`
	DataType         string          `json:"data_type"`
	HeadShape        []int           `json:"head_shape"`
	Heads            int             `json:"n_heads"`
	Block            int             `json:"matmul_block_size"`
	SelectFirstToken bool            `json:"select_first_token"`
	Transposed       *bool           `json:"is_transposed"`
}
type Task struct {
	EvaluationKeys []KeyRequirement        `json:"evaluation_keys"`
	UseBTP         bool                    `json:"use_btp"`
	PackStyle      string                  `json:"pack_style"`
	MatrixStyle    string                  `json:"mat_pack_style"`
	BlockShape     []int                   `json:"block_shape"`
	Inputs         map[string]FeatureParam `json:"task_input_param"`
	Outputs        map[string]FeatureParam `json:"task_output_param"`
	Heads          int                     `json:"n_heads"`
	Block          int                     `json:"matmul_block_size"`
}
type keyState struct {
	Params ckks.Parameters
	SK     *rlwe.SecretKey
	PK     *rlwe.PublicKey
}
type Client struct {
	Task      Task
	Keys      keyState
	plan      clientcontext.KeyPlan
	context   *clientcontext.Context
	encoder   *ckks.Encoder
	encryptor *rlwe.Encryptor
	decryptor *rlwe.Decryptor
}

func planFor(b bool) clientcontext.KeyPlan {
	return clientcontext.KeyPlan{DefaultRotations: true, Bootstrapping: b}
}

// NewKeyClient validates an explicit v6 profile before any key generation.
func NewKeyClient(data []byte, bootstrap bool) (*Client, error) {
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}
	if len(entries) != 1 {
		return nil, errors.New("key generation requires one parameter entry")
	}
	for id := range entries {
		params, err := parseTaskParameters(data, id)
		if err != nil {
			return nil, err
		}
		return &Client{Keys: keyState{Params: params}, plan: planFor(bootstrap)}, nil
	}
	return nil, errors.New("missing parameters")
}

func NewClient(task, params []byte) (*Client, error) {
	if len(task) == 0 {
		return NewKeyClient(params, true)
	}
	c := new(Client)
	if err := json.Unmarshal(task, &c.Task); err != nil {
		return nil, err
	}
	if len(c.Task.Inputs) == 0 || len(c.Task.Outputs) == 0 {
		return nil, errors.New("task requires inputs and outputs")
	}
	if c.Task.EvaluationKeys == nil {
		return nil, errors.New("task is missing evaluation_keys; regenerate it with the current gen_mega_ag.py")
	}
	if c.Task.PackStyle != "ordinary" && c.Task.PackStyle != "multiplexed" {
		return nil, errors.New("invalid pack_style")
	}
	names := make([]string, 0, len(c.Task.Inputs))
	for n := range c.Task.Inputs {
		names = append(names, n)
	}
	sort.Strings(names)
	p, err := parseTaskParameters(params, c.Task.Inputs[names[0]].ParameterID)
	if err != nil {
		return nil, err
	}
	c.Keys.Params, c.plan = p, planFor(c.Task.UseBTP)
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(params, &entries); err != nil {
		return nil, err
	}
	for id := range entries {
		if err := requireMatchingTaskParameters(params, []string{id}, p); err != nil {
			return nil, err
		}
	}

	for _, fs := range []map[string]FeatureParam{c.Task.Inputs, c.Task.Outputs} {
		for n, p := range fs {
			if err := requireMatchingTaskParameters(params, []string{p.ParameterID}, c.Keys.Params); err != nil {
				return nil, err
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
			if p.Dim < 0 || p.Dim > 2 || p.Channel < 1 || p.Level < 0 || p.Level > c.Keys.Params.MaxLevel() {
				return nil, errors.New("invalid feature dimensions or level")
			}
			if p.Dim > 0 && (len(p.Shape) != p.Dim || product(p.Shape) <= 0) {
				return nil, errors.New("invalid feature shape")
			}
			fs[n] = p
		}
	}
	if err := c.requireTaskKeys(c.Task.EvaluationKeys); err != nil {
		return nil, err
	}
	return c, nil
}

// BindTask validates a task without replacing the generated or restored keys.
func (c *Client) BindTask(task, params []byte) error {
	bound, err := NewClient(task, params)
	if err != nil {
		return err
	}
	if !bound.Keys.Params.Equal(&c.Keys.Params) || bound.plan.Bootstrapping && !c.plan.Bootstrapping {
		return errors.New("task parameters or evaluation-key plan do not match client profile")
	}
	if err := c.requireTaskKeys(bound.Task.EvaluationKeys); err != nil {
		return err
	}
	c.Task = bound.Task
	return nil
}

func (c *Client) initCrypto() {
	c.encoder = ckks.NewEncoder(c.Keys.Params)
	if c.Keys.PK != nil {
		c.encryptor = ckks.NewEncryptor(c.Keys.Params, c.Keys.PK)
	}
	if c.Keys.SK != nil {
		c.decryptor = ckks.NewDecryptor(c.Keys.Params, c.Keys.SK)
	}
}

// GenerateKeys creates the single client key pair without evaluation keys.
func (c *Client) GenerateKeys() error {
	ctx, err := clientcontext.New(c.Keys.Params, c.plan)
	if err != nil {
		return err
	}
	c.context = ctx
	c.Keys.SK, c.Keys.PK = ctx.SecretKey, ctx.PublicKey
	c.initCrypto()
	return nil
}

// StartSetup streams both files with disk acknowledgements and length patches.
func (c *Client) StartSetup() (*OutputStream, error) {
	if err := c.GenerateKeys(); err != nil {
		return nil, err
	}
	ctx := c.context
	return startOutput(func(s *OutputStream) error {
		if err := s.output("eval", ctx.WriteEval); err != nil {
			return err
		}
		return s.output("full", ctx.WriteFull)
	}), nil
}

func (c *Client) WriteFull(sink clientcontext.Sink) (uint64, error) {
	if c.context == nil {
		return 0, errors.New("setup or restore required")
	}
	return c.context.WriteFull(sink)
}

func (c *Client) Restore(data []byte) error {
	ctx, err := clientcontext.Restore(data, c.plan)
	if err != nil {
		return err
	}
	if !ctx.Parameters.Equal(&c.Keys.Params) {
		return errors.New("full context parameters do not match task")
	}
	c.context = ctx
	c.Keys.SK, c.Keys.PK = ctx.SecretKey, ctx.PublicKey
	c.initCrypto()
	return nil
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

// StartRestore reads the small client-only container from bounded OPFS input.
func (c *Client) StartRestore(size uint64) *OutputStream {
	return startOutput(func(s *OutputStream) error {
		source := &inputSource{stream: s, id: "full", size: size}
		ctx, err := clientcontext.RestoreFrom(source, size, c.Keys.Params, c.plan)
		if err != nil {
			return err
		}
		c.context = ctx
		c.Keys.SK, c.Keys.PK = ctx.SecretKey, ctx.PublicKey
		c.initCrypto()
		return nil
	})
}
