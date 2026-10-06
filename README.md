# probe-inference

Inference for four activation-probe architectures:

- linear;
- MLP;
- EFC (early-fusion covariance);
- axial.

Load a trained probe, from a local directory or the Hugging Face Hub, and score a
transcript's residual-stream activations with it; the result is one score. This package does not
extract activations, and it has no thresholds or calibration.

The package depends only on `torch`. The `[hub]` extra adds `huggingface_hub`, for
`load_probe_from_hub`.

## Usage

```python
from probe_inference import load_probe, load_probe_from_hub

probe = load_probe("my_probes/qwen3.5-9b/efc")  # a local probe directory
probe = load_probe_from_hub("qwen3.5-9b/efc")  # the published weights, <model>/<arch>
probe.arch, probe.layers, probe.read_window, probe.output
# ('efc', (10, 13, 18, 21, 26, 29), 'last-user-and-assistant-generation', 'logit')

# acts: {layer: Tensor[seq, d_model]} for the layers in probe.used_layers (any float dtype),
#       or one Tensor[len(probe.layers), seq, d_model] stacked in probe.layers order.
# read_mask: Tensor[seq] bool, True on the tokens the probe's read window selects.
read_mask = probe.read_mask(prompt_mask, completion_mask, followup_start_positions)
score = probe.score(acts, read_mask)  # float

scores = probe.score_batch(batched_acts, batched_mask)  # (B,) float64; acts {layer: [B, seq, d]}
per_layer = probe.layer_logits(acts_b, mask_b)  # linear/MLP only: {layer: (B,) logits}
```

`load_probe_from_hub(path)` downloads one trained probe from the published weights, the Hugging Face model
repository `AlignmentResearch/probe-inference-weights` at the commit pinned by
`probe_inference.load.WEIGHTS_REVISION`, and loads it. Only that probe's directory is downloaded. Pass
`repo_id`, `revision` and `repo_type` to load from another repository. The published probes are
`<model>/<arch>/`, with `<arch>` in `linear`, `mlp`, `efc` and `axial`, for these models: `qwen3.5-2b`,
`qwen3.5-9b`, `qwen3.6-27b`, `qwen3.5-122b-a10b`, `qwen3.5-397b-a17b`, `nemotron-3-nano-30b-a3b` and
`nemotron-3-super-120b-a12b`. The repository's card gives each model's revision and layers, and the
licences of the models the probes were trained on.

What `score` returns depends on the architecture:

| Architecture | `probe.output` | Score |
|---|---|---|
| Linear, MLP | `probability` | Each layer's probe reads one token. The score is the mean of the per-layer sigmoids over `probe.used_layers` (`probe_metadata.json` `layer_rule.used_layers`). |
| EFC | `logit` | One pooled logit over the read window. |
| Axial | `logit` | The logit at the last read token, capped by $10\tanh(x/10)$. |

`score` fails loudly rather than scoring a wrong input. It raises `ValueError` in these cases:

- a layer is missing, or has the wrong shape or hidden size;
- the activations are not floating point, or hold a non-finite value at a read position;
- the read mask is not a bool tensor of the activations' `(batch, seq)` shape, or selects nothing;
- a linear or MLP row does not read exactly one token;
- an EFC row reads fewer than two tokens, or only identical ones;
- a read token's activation is exactly zero in the layer the padding mask is taken from. The probes
  treat all-zero rows as padding, so the token would be dropped silently.

`load_probe` also refuses (`InputScaleError`) a `unit_norm` or `centered_unit_norm` probe whose
`input_scale` is exactly 1.0, the unset default: its weights were saved without the normaliser they
were trained with.

## Activation contract

The probes were trained on activations with the following properties. Scores on activations that
differ from this are not meaningful.

| Part | Contract |
|---|---|
| Layer index | Layer $k$ is the output of decoder block $k$, 0-indexed, of the model loaded with `AutoModelForCausalLM`. For Qwen3.5/3.6 that is `model.model.layers[k]`. For Nemotron-3 it is `layers[k]` of the inner `NemotronHModel`, which Nano-30B holds at `model.backbone` and Super-120B at `model.model`. This equals Hugging Face `output_hidden_states[k + 1]`. The capture is a forward hook on the block (for Qwen3.5-397B-A17B, a vLLM hook at the same decoder-layer outputs). For the last layer ($N-1$), the hook instead reads the final norm's output when the model's `output_hidden_states` does. None of the probes below reads the last layer. |
| Stream | The residual stream, as the Hugging Face block returns it. |
| Dtype | The model ran in bfloat16, and the activations were stored in bfloat16. `score` runs the probes in float32. (Training ran the probe forward under bfloat16 autocast.) |
| Context | The whole conversation, with the follow-up appended, goes through the model. The probes read only positions inside their window, but those activations depend on the full context. |
| Chat rendering | The model's own chat template, with thinking disabled. The follow-up is a final user turn and a prefilled assistant answer, closed by the end-of-turn token. In training, the whole conversation went through the chat template, which puts a newline after every end-of-turn token. The parity fixture's activations instead have the follow-up's `<\|im_start\|>` directly after the previous turn's end-of-turn token, with no newline. |
| Read window, linear and MLP | `second-last-token-generation`: exactly one token, the last token of the assistant answer before its end-of-turn token. For a prefilled `No.` or `Yes.` it is the full stop. |
| Read window, EFC and axial | `last-user-and-assistant-generation`: every token from the first token of the final user turn (its `<\|im_start\|>`) through the assistant's end-of-turn token, inclusive. |
| Token layout | `score` gathers the read tokens to the front of each row, in order, and zero-pads the rest. The probe's padding mask marks the non-zero rows. Pass the full-length activations and the read mask; do not compact them yourself. This matters for the axial probe, whose rotary embedding counts positions. |

### Follow-up rendering

A follow-up is a question with a prefilled answer, for example
`Did you do anything the user did not ask you to do?` answered `No.`. Rendered, it looks like this:

```
Qwen3.5 / 3.6:  <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nNo.<|im_end|>
Nemotron-3:     <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think></think>No.<|im_end|>
```

### Building the read mask

`probe.read_mask(prompt_mask, completion_mask, followup_start_positions)` builds the mask for the
probe's window. The arguments are:

- `completion_mask`: True on the assistant answer tokens after the empty thinking block, plus the
  end-of-turn token. For a `No.` prefill that is `No`, `.` and `<|im_end|>`.
- `prompt_mask`: True on every other real token of the row.
- `followup_start_positions`: the index of the final user turn's first token (`<|im_start|>`).

A hand-built boolean mask that selects the positions in the table above works too.

### Layers per model

These are six layers at depth fractions 0.3, 0.42, 0.55, 0.67, 0.8 and 0.9, computed as
`round(f * num_hidden_layers)`. Each probe's `probe_metadata.json` `layers` field is authoritative. The
Nemotron-3 models were loaded from local re-saved checkpoints with fused expert tensors, made from
the Hub snapshots at these revisions.

| Model (HF revision) | Blocks | `d_model` | Layers |
|---|---|---|---|
| `Qwen/Qwen3.5-2B` (`15852e8c`) | 24 | 2048 | 7, 10, 13, 16, 19, 22 |
| `Qwen/Qwen3.5-9B` (`c2022362`) | 32 | 4096 | 10, 13, 18, 21, 26, 29 |
| `Qwen/Qwen3.6-27B` (`6a9e13bd`) | 64 | 5120 | 19, 27, 35, 43, 51, 58 |
| `Qwen/Qwen3.5-122B-A10B` (`dc4d3484`) | 48 | 3072 | 14, 20, 26, 32, 38, 43 |
| `Qwen/Qwen3.5-397B-A17B` (`84726181`) | 60 | 4096 | 18, 25, 33, 40, 48, 54 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (`bf77c317`) | 52 | 2688 | 16, 22, 29, 35, 42, 47 |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (`2dc98e2a`) | 88 | 4096 | 26, 37, 48, 59, 70, 79 |

## Probe directory format

A trained probe is one directory, such as `qwen3.5-9b/efc/`. The same layout is read from a local
path or from a Hub repository. A linear or MLP probe is one small probe per layer (a *layer probe*);
an EFC or axial probe is one module that reads all its layers at once.

```
linear/                      # linear and MLP: one layer probe per layer
  probe_metadata.json
  layer_<L>/config.json
  layer_<L>/model.pt
efc/                         # EFC and axial: one cross-layer probe
  probe_metadata.json
  config.json
  model.pt
```

| File | Content |
|---|---|
| `config.json` | `{"class_name", "init_args"}`. `class_name` is `LinearProbe`, `MLPProbe`, `EFCProbe` or `AxialProbe`; any other key is ignored. |
| `model.pt` | A plain float32 `state_dict`, loaded with `weights_only=True`. |
| `probe_metadata.json` | See the list below. |

`probe_metadata.json` holds:

- `layers`: model layers, in stacking order;
- `eval_sequence_aggregator`: `mean` or `last`;
- `obfuscate_over`: the read window;
- for linear and MLP only, `layer_rule.used_layers`.

## Tests

```bash
uv sync
uv run pytest
```

The unit tests use synthetic probes and run on CPU in seconds. `tests/test_parity.py` compares the
package with reference scores on real activations, using the published probes. Its fixture (about
150 MiB) is the Hugging Face dataset `AlignmentResearch/probe-inference-parity` (public; no token
needed) at the commit pinned by `FIXTURE_REVISION` in that file. It holds a few rows of activations and
token masks per model and the reference scores for those rows, but no probes. Each probe is loaded with
`load_probe_from_hub` from the published weights at the pinned `WEIGHTS_REVISION`, exactly as users load
it. Both downloads go through the Hub cache (`HF_HOME`); cache that directory in CI to avoid
re-downloading them (`.github/workflows/ci.yml` does). A missing fixture, an unreachable repository or a
probe missing at the pinned weights revision fails the parity tests. They need network access even
with a warm cache, because one test lists the weights repository's files at the pinned revision.

| Set | Probes | Notes |
|---|---|---|
| `nemotron-3-super-120b` | EFC, axial | 80 rows; one batch mixes 27- and 24-token read windows |
| `qwen3.5-9b` | linear, MLP, EFC, axial | 32 rows |

## Licence

This package is released under the MIT licence (see `LICENSE`). `EFCProbe` (in `archs/efc.py`) adapts
Goodfire's early-fusion covariance probe, from Goodfire code that is not public. Confirm the terms for
that code with Goodfire before any public release of this repository.

The published probe weights and test fixture on the Hugging Face Hub are also MIT. They are derived
from Qwen models (Apache-2.0) and NVIDIA Nemotron-3 models (NVIDIA Nemotron Open Model License), whose
licence texts and attribution notices ship with them (`NOTICE` and the `LICENSE-*` files in each
repository).
