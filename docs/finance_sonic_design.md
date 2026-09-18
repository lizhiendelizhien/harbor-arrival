# Financial Sonic: Monthly Encoder and Decoder

## Scope and Architecture

This implements conditional future-trajectory reconstruction for one equity at
each monthly anchor. It preserves the reference method's plain MLP encoder,
shared continuous latent, FSQ auxiliary branch, and causal Transformer with a
32-step KV cache. The PPO runner trains low-level reference tracking with the
retained Gaussian Actor and value Critic in either one process or synchronous
multi-GPU data parallel mode. There is no portfolio action, trading-profit reward,
high-level policy, or causal latent predictor.

```text
future descriptors [B,S,10,15]
    -> robust normalization (also retained as the clean Kin target)
    -> stored training noise + semantic masking (clean during evaluation)
    -> flatten last two axes
    -> MLP: 150 -> 2048 -> 1024 -> 512 -> 512 -> 64 (SiLU)
    -> reshape to two 32-dimensional tokens
         |
         +-> raw continuous z [B,S,64]
         |     + current state [B,S,16], normalized separately
         |     -> Transformer input [B,S,80]
         |     -> 6 layers, width 256, 4 heads, FFN 1024, RoPE, window 32
         |     -> 10 normalized monthly log returns
         |     -> inverse scaling -> monthly and cumulative log returns
         |
         +-> FSQ: 32 scalar levels per dimension, no joint indices
               -> kin MLP: 64 -> 2048 -> 1024 -> 512 -> 512 -> 150
               -> reconstruct normalized future descriptors [B,S,10,15]
               -> SAME encoder -> reencoded z [B,S,64]
```

There is one encoder output, with raw and quantized routes, rather than two
independently learned latent heads. `kin` gets only quantized tokens, with no
current-state shortcut. Re-encoding does not quantize, mask, add noise, or detach
the cycle path.
`dyn` gets the raw latent, without tanh or quantization. The original Transformer
implementation is reused without changes; its action projection now predicts
monthly returns. The current baseline does not instantiate `s_pred` or `z_gmm`.

Decision on 2026-09-08: omit BOTH `s_pred` and `z_gmm`, their auxiliary losses,
and GMM-based deployment latent safety filtering. This is an explicit exception
to Stage5, not an accidental omission. Encoder/FSQ/kin routing and the 32-step
Transformer window remain unchanged. The only learned dyn output is the
ten-dimensional trajectory action. The cumulative path is derived by summation,
not another prediction head.

## Data and Feature Definitions

Input is `data/us_socket/`'s cleaned `monthly_canonical.csv`. Use paired valid raw
and QFQ rows, QFQ OHLC for relative prices, raw volume, and raw close for a uniform
dollar-volume proxy `D_t = raw_close_t * volume_t`. This is an approximate monthly
liquidity measure, not actual traded notional. Reported amount/turnover are not
required; switching between reported amount and a fallback is avoided.

Define `C,O,H,L` as QFQ prices, `r_t=C_t/C_(t-1)-1`, `m_j=C_t/C_(t-j)-1`,
`v_j=population_std(r_(t-j+1),...,r_t)`, and `eps=1e-6`.
The market proxy `b_t` is the same-month median simple return of the available
valid universe; `M6_t=product(1+b_u)-1` over the latest six consecutive months.
It is a descriptive universe statistic, not a tradable index or a total-return
benchmark. It is built from the full supplied file even when selecting one stock.

Current state order (one state as of the completed month t):

| Index | Feature | Formula |
| --- | --- | --- |
| 0 | return_1m | r_t |
| 1 | gap_return | O_t / C_(t-1) - 1 |
| 2 | intramonth_return | C_t / O_t - 1 |
| 3 | amplitude | (H_t-L_t) / O_t |
| 4 | close_location | (C_t-L_t)/(H_t-L_t), or 0.5 for a flat bar |
| 5 | momentum_3m | m_3 |
| 6 | momentum_6m | m_6 |
| 7 | momentum_acceleration | log1p(m_3)/3 - log1p(m_6)/6 |
| 8 | volatility_3m | v_3 |
| 9 | volatility_6m | v_6 |
| 10 | volatility_ratio | log((v_3+eps)/(v_6+eps)) |
| 11 | drawdown_6m | C_t / max(C_(t-5),...,C_t) - 1 |
| 12 | liquidity_rank | Same-month percentile of D_t, ties get average rank |
| 13 | volume_surprise_3m | log(V_t / mean(V_(t-3),...,V_(t-1))) |
| 14 | market_return_1m | b_t |
| 15 | relative_strength_6m | m_6 - M6_t |

All rolling descriptors are computed as of t. They do not turn the input into a
stack of historical states. Seven consecutive monthly bars are needed to compute
six monthly returns. Zero-volume or invalid rows create gaps, not zero signals.

Future descriptor order, for each k=1,...,10 and u=t+k:

| Index | Feature | Formula |
| --- | --- | --- |
| 0 | forward_log_return | log(C_u/C_(u-1)) |
| 1 | cum_log_return | log(C_u/C_t) |
| 2 | gap_return | O_u/C_(u-1)-1 |
| 3 | intramonth_return | C_u/O_u-1 |
| 4 | amplitude | (H_u-L_u)/O_u |
| 5 | close_location | (C_u-L_u)/(H_u-L_u), or 0.5 |
| 6 | momentum_3m_delta | m3_u-m3_t |
| 7 | momentum_6m_delta | m6_u-m6_t |
| 8 | volatility_3m_ratio | log((v3_u+eps)/(v3_t+eps)) |
| 9 | volatility_6m_ratio | log((v6_u+eps)/(v6_t+eps)) |
| 10 | running_drawdown | C_u/max(C_t,...,C_u)-1 |
| 11 | volume_surprise | log(V_u/mean(V_(u-3),...,V_(u-1))) |
| 12 | dollar_volume_surprise | log(D_u/mean(D_(u-3),...,D_(u-1))) |
| 13 | relative_strength_1m | r_u-b_u |
| 14 | relative_strength_6m | m6_u-M6_u |

The boolean mask has shape `[B,S,10,15]`, is separate from the 150 continuous
values, and is not a reconstruction target. The supplied sequence builder emits
complete windows with all-valid masks. The model also supports partial feature
masks, filling masked normalized inputs with zero and excluding them from losses.
Entirely missing windows are rejected. Future volume baselines are relative to
each future month, while momentum and volatility changes are anchored at t.

### Encoder Denoising

Training adds independent uniform noise in `[-0.05,0.05]` normalized feature units,
then zeros hidden and invalid entries. The noise is added after normalization and
clipping, so visible encoder inputs can extend to `[-10.05,10.05]`; clean Kin
targets stay clipped to `[-10,10]`. Four nested semantic modes use weights
`[1,1,1,0.1]`, normalized to probabilities `[10/31,10/31,10/31,1/31]`:

| Mode | Newly Hidden Fields | Visible Count |
| --- | --- | --- |
| 0 | None | 15 |
| 1 | Volume and dollar-volume surprise (11,12) | 13 |
| 2 | Also relative strength (13,14) | 11 |
| 3 | Also volatility ratios and running drawdown (8,9,10) | 8 |

A mode is shared across all ten future months. Each environment holds it for a
uniformly sampled 2-5 monthly anchor steps, then resamples; episode reset also
resamples. This is a financial adaptation of Stage5's persistent selectors, not
an assertion that 2-5 monthly steps equal its 2-5 simulation seconds. Noise is
refreshed at each new anchor. A separate RNG preserves reference-clip sampling.

`encoder_mask_type [E,1]` and `encoder_noise [E,10,15]` are cached actor-only
observation fields, not extra MLP inputs. Repeated observation reads are pure.
Rollouts, PPO epochs, and observation-history cache rebuilds consume the same
stored values. Model forwards never resample or infer augmentation from
`model.training`, since rollout runs in evaluation mode while PPO runs in
training mode.

The clean `future_reference`, data-validity `future_mask`, current state, critic,
and reward targets are unchanged. Kin supervises every valid clean field,
including hidden fields. Cycle directly re-encodes normalized Kin output without
another normalization or corruption. Model calls omitting both augmentation
arguments use clean inputs; checkpoint evaluation always uses this path.

Reference Kin ordering correction on 2026-09-09: the robot clean target producer
still packs semantic blocks into 440 values. Its loss now splits those blocks,
reshapes each by frame, and concatenates per-frame features. Finance remains
consistently `[month, feature]` and does not copy the former global-reshape bug.
Robot tensor/checkpoint shapes do not change, but resumed Kin training now has
a corrected objective; previously learned incorrectly packed outputs may need
retraining.

## Normalization and Losses

For stage-one reference playback, use the entire supplied reference pool without
a train/validation/test partition (decision on 2026-09-08). Call `fit_normalizers`
on that pool, or load `normalization_stats.json` using `load_normalizers`.
For each feature, subtract its reference-pool median and divide by its
reference-pool IQR; use scale 1 for nearly constant features. Clip normalized inputs
to [-10,10]. Statistics are fixed buffers saved in `state_dict`; forward and
evaluation do not update them. Kin reconstructs this normalized/clipped target;
`kin_reconstruction` converts it back to feature units. Dyn outputs are inverse
scaled with the future monthly-log-return statistics. Dyn supervision uses the
original monthly log returns, not a clipped price target.

The retained `financial_sonic_loss` is a supervised diagnostic/baseline helper,
not the PPO training objective. It returns:

- `dyn`: masked Smooth L1 on monthly log returns divided by their reference-pool scale.
- `cumulative`: masked Smooth L1 on cumulative sums, scaled by sqrt(k) times
  the same return scale. A missing monthly return invalidates that and all later
  cumulative horizons for that anchor.
- `kin`: masked MSE between normalized future descriptors and kin output.
- `cycle`: MSE between shared encoder's raw z and its re-encoding of kin output.
- `total = dyn + cumulative + 0.01*kin + cycle` by default. Coefficients are
  configurable initial settings, not claims of optimal financial weighting.

All three network branches remain differentiable; FSQ uses straight-through
gradients. There is no stop-gradient on the original cycle latent. Optional
`burn_in` excludes initial anchor steps from all supervised loss terms while
preserving their role as attention context.

## PPO Tracking Training

`policy.py` supplies financial backbones to the existing `Actor` and `Critic`
wrappers. The actor has no running observation normalizer; fixed reference-pool
feature scaling happens inside the financial observation/model interface. The
critic retains the original running mean/std normalizer and an MLP with widths
`[2048,2048,1024,1024,512,512]`, SiLU, and one value output.

Each vector environment plays a uniformly sampled complete reference clip.
The default clip is 64 consecutive monthly anchors; short remaining tails are
dropped by the existing sequence builder. Terminal clips reset independently
to sampled clips, clear histories, and have no timeout bootstrap. Market states
advance from the dataset independently of the action. This preserves a PPO
reference-tracking training structure, not controllable physical dynamics.

The actor receives only the current `[16]` state and the encoded future latent.
The separate privileged critic observation has 410 dimensions:

```text
10 chronological current-state history slots * 16 = 160
10 chronological previous-action history slots * 10 = 100
current clean future reference * (10 * 15) = 150
total = 410
```

State history includes the current month; action history excludes the action
about to be sampled. Missing history at reset is zero-padded. Neither history
stack is concatenated into the actor's 16-dimensional current-state input.

The Transformer returns normalized action means. The original Gaussian Actor
samples ten normalized monthly returns with a learned std initialized to 0.05
and clamped to `[0.001,0.5]`. The model's `normalized_actions` output exposes
these units; `log_returns = normalized_actions * return_scale + return_center`.
The same conversion applies to sampled actions before interpreting price paths.

The low-level reward uses original, unclipped monthly log-return targets:

```text
y_k = (reference_log_return_k - return_center) / return_scale
e_k = action_k - y_k
r_monthly = exp(-mean_k(e_k**2))
r_path = exp(-mean_k((cumsum(e)_k / sqrt(k))**2))
r_change = exp(-mean_k=2..10((e_k - e_(k-1))**2 / 2))
r_vol = exp(-(std_population(action) - std_population(y))**2)
reward = 0.30*r_monthly + 0.40*r_path + 0.20*r_change + 0.10*r_vol
```

This task-specific reward is maximal at exact reference tracking. It is neither
PnL nor a stock-selection reward. Reward error arithmetic uses float64 to avoid
overflow on finite extreme inputs; network tensors and rewards are float32.

This is `financial_tracking_v1`, specified in
[Financial Reward V1](superpowers/specs/2026-09-08-finance-reward-v1-design.md).
All kernel widths are fixed at one in normalized units. Population standard
deviation uses denominator ten. Change matching penalizes deviations from real
changes, not absolute movement; genuine crashes and flat references can both
score one when reproduced exactly. These initial weights are not calibrated
claims of optimal financial performance.

Rolling consistency compares current `action[:9]` to previous `action[1:]` in
the same uninterrupted clip. It remains diagnostic-only (`lambda_roll=0`), with
no automatic activation. First steps after any reset are excluded from its
aggregates; empty aggregates are JSON null with valid count zero. The previous
overlap tracking error is also logged to distinguish revisions of wrong older
predictions. All these metrics score sampled PPO actions; exploration noise is
therefore included. They do not directly measure deterministic-policy jitter.

Iteration logs retain the individual squared errors, component rewards and
weighted contributions, plus component quantiles and saturation fractions.
Error aggregation stays float64 through JSON conversion. Cumulative RMSE and
direction agreement at 1/3/6/10 months use inverse-scaled original log returns,
not the signs of centered normalized actions. Kin/cycle remain separate losses.

The actual optimizer objective is:

```text
L = L_clipped_policy + L_clipped_value - 0.01 * entropy
  + 0.01 * L_kin + 1.0 * L_cycle
```

There is no `s_pred`, `z_gmm`, direct supervised dyn Huber, or extra cumulative
Huber term in PPO. `financial_auxiliary_losses` supplies only kin and cycle.
Rollout skips kin/FSQ reconstruction; PPO re-encodes observations with the live
encoder so policy gradients update it as well as the dynamic decoder.

`ppo.py` mirrors the reference trainer's effective single-process update math:
24 rollout steps, five epochs, four environment-stream minibatches, gamma 0.99,
lambda 0.95, policy/value clipping 0.2, value coefficient 1, and gradient norm
limit 0.1. GAE uses sample-std advantage normalization. Value loss has no extra
factor of 0.5. AdamW uses one actor/critic learning rate starting at `2e-5`,
betas `(0.9,0.999)`, epsilon `1e-8`, and weight decay 0. KL adaptation uses
target 0.01, multiplier/divisor 1.5, and bounds `[1e-5,2e-4]`. The reference YAML's
separate `critic_learning_rate` is not used by its inherited optimizer setup.
For a singleton update, critic variance statistics are left unchanged rather
than evaluating an undefined sample variance.

Before rollout, snapshot up to 31 detached KV history slots. Reuse that frozen
prefix across PPO epochs, preserving whole time sequences when shuffling
environment minibatches. All 24 new rollout steps contribute to losses; they
are not discarded as a 31-step burn-in. Episode masks prevent access across
resets. Live rollout caches persist across PPO updates, as in the reference
trainer; this is distinct from restarting an independent inference session.

Reference sources are `gear_sonic/trl/trainer/ppo_trainer.py`,
`ppo_trainer_aux_loss.py`, `config/algo/ppo_im_phc.yaml`,
`config/algo/trl/ppo.yaml`, and `sonic_release_stage5.yaml` under the reference
repository. The retained MLP, Transformer, Actor and Critic are reused. The
financial playback environment, reward, observation scaling and critic fields
are domain mappings. Semantic masking and clean-target denoising are ported with
the financial groups and durations above. Physical disturbances and
terrain/failure curricula are not applicable to the financial environment. The
finance runner supports true multi-rank training through explicit reference/env
shards, gradient all-reduce, and rank-zero checkpoint publication; this is not a
claim of full simulator/trainer parity.

## Time Axes, Reference Pool, and Cache

`S` counts successive historical anchors; `H=10` counts each anchor's forecast
horizon. One anchor produces an entire ten-month return path. H is never fed as
ten successive Transformer cache steps. Each layer's 32-step attention window
includes the current step and up to 31 prior steps; deeper states can indirectly
carry information from earlier times.

`iter_sequences` emits non-overlapping blocks from a single symbol, rejects gaps,
and drops short tails. Optional date bounds restrict a selected reference range
and must contain every anchor's complete future horizon. They do not impose a
train/validation/test partition. The default smoke check uses one full pool.
For independent inference, reset cache between independently sampled blocks,
symbols, time discontinuities, and model-weight changes. The PPO cache lifecycle
is described above. In a batch, each stream slot must retain its symbol or
be reset using `reset_mask`. Use `episode_attnmask` when explicitly packing
multiple episodes; True entries block attention between those positions.

Teacher latents contain future information. A causal attention mask does not
make such latents or their cached history deployable forecasts. This module
implements privileged representation learning. Actual forecasting evaluation
requires a later causal latent source and caches built with that same source;
high-level latent selection cannot control realized market prices.

Canonical QFQ data and the supplied universe retain the source dataset's
point-in-time adjustment and listing-history limitations. Requiring complete
future windows also conditions on future data availability. Reconstruction
coverage is not an eligible investment universe or evidence of tradable returns.

## Usage

Install with `python -m pip install -e '.[finance]'`; install pytest separately
for tests. The MLP path does not import torchvision; only selecting ResNet needs it.
Actor/Critic imports do not require W&B; that dependency is loaded only by the
W&B utility itself.

Run a bounded PPO check first (choose a new output directory):

```bash
python -m scripts.train_finance_sonic --canonical PATH_TO_MONTHLY_CANONICAL \
  --symbols AAPL MSFT --output-dir /tmp/finance-ppo-example --tiny \
  --num-envs 4 --iterations 3 --epochs 1 --num-minibatches 2
```

Omit `--tiny` to use full original network widths. The local CLI defaults to 16
environments rather than the reference experiment's 4096; `--num-envs` controls
resource use. `--iterations` is required and counts additional updates on resume.
No implicit date split is created. `--start`/`--end` narrow the reference pool;
`--normalization` loads exported statistics instead of fitting the selected pool.

For a true four-GPU run over the reconstructed trajectory archive:

```bash
TASK=finance_sonic_monthly \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 NUM_ENVS=1024 SEED=42 \
FINANCE_MULTI_GPU_LAUNCHER=torchrun \
bash scripts/train_finance_sonic_4gpu.sh
```

The launcher accepts either the archive root or its `trajectories/` child and
resolves `trajectory_index.csv`, `metadata.pkl`, and `manifest.json` from the
root. `NUM_ENVS` is per rank, matching the Sonic launcher convention, so the
example runs 4096 environment streams globally. `torchrun` assigns one process
to each visible device. Rank `r` owns the round-robin reference shard
`sequences[r::world_size]` and independent environment RNG streams. Every PPO
minibatch averages coalesced Actor, Critic, Kin, and Cycle gradients across all ranks;
advantage moments, adaptive-KL input, Critic running statistics, and reported
metrics are also pooled. Only rank 0 writes the immutable run recipe and the one
checkpoint for each save step. The checkpoint's distributed metadata explicitly
records `reference_sharding=round_robin` and `environment_sharding=per_rank`;
`dataset_partition=none` still means no train/validation split.

The account launching the job needs read permission on all three companion files
and the `trajectories/*.pkl` files. The launcher performs an early readability
check for the companion files and reports the exact path when permissions are
insufficient.

Archive startup is intentionally visible but not instantaneous. The current
archive contains 23,326 segment files; each torchrun rank validates the shared
catalog and builds its reference pool independently before reference/env
sharding. The trainer emits `archive_load_start` and periodic
`archive_load_progress` events before the normal `start` event. A cold full
archive load can take roughly two minutes on network storage, so use the
following bounded smoke first when checking a new node or environment:

```bash
SYMBOLS=AAPL MAX_ITERATIONS=1 SAVE_INTERVAL=1 NUM_ENVS=4 TINY=1 \
ROLLOUT_STEPS=2 EPOCHS=1 NUM_MINIBATCHES=1 \
bash scripts/train_finance_sonic_4gpu.sh
```

The launcher log is written under `/tmp/finance_launch_logs`; `tail -f` that
file while waiting for the first rank events. A full run without `SYMBOLS`
reads all indexed segments on every rank by design; GPU memory and utilization
should also be checked before sharing devices with another training job.

The launcher uses the current `python` only when it is suitable for the
project. If `PYTHON_BIN` is unset, it probes the project and nearby virtual
environments and prefers the first interpreter that can import all finance
dependencies. Set `PYTHON_BIN` explicitly when the environment lives
elsewhere, for example:

```bash
PYTHON_BIN=/path/to/finance-venv/bin/python \
PYTHONPATH=/path/to/extra/site-packages \
bash scripts/train_finance_sonic_4gpu.sh
```

The launcher also appends `FINANCE_PYTHONPATH` (or `LOCAL_PYTHON_DEPS`) to
`PYTHONPATH` before probing. When neither is set, existing `.python_deps` and
`/tmp/finance-sonic-deps` bundles are used automatically if present. This lets
the common AeroStep torch environment reuse the finance-only packages without
modifying that external virtualenv.

For a new environment, install the project extras into that same interpreter:

```bash
/path/to/finance-venv/bin/python -m pip install -e '.[finance]'
```

On a real run the launcher checks `torch`, `numpy`, `omegaconf`, `tensordict`,
`vector_quantize_pytorch`, and (when `LOGGER=tensorboard`)
`torch.utils.tensorboard` before starting `torch.distributed.run`. Dry-runs
skip the required-dependency preflight and training launch, while the candidate
probe may still invoke lightweight interpreter import checks so the resolved
Python path remains meaningful. TensorBoard is included in the `finance` extra;
use `LOGGER=none` only when a dependency-free JSON-log run is intentional.

TensorBoard follows the table-tennis runner's layout and rank ownership. A
training run writes event files under `OUTPUT_DIR/tensorboard`, and only global
rank 0 creates the writer. Scalars use the restored PPO iteration as their
global step, so a resumed run continues at the checkpoint iteration instead of
starting a second step-zero curve. The writer flushes every ten seconds and is
explicitly flushed/closed at checkpoint and process boundaries.

The main scalar groups are `Loss/` (`ppo_loss`, `value_loss`, `kin`, `cycle`),
`Policy/` (`entropy`, `kl`, `grad_norm`, `learning_rate`), `Reward/` (monthly,
path, change, volatility and contribution terms), and `Tracking/` (MSE, RMSE,
direction accuracy and rolling diagnostics). These values are the already
all-reduced metrics returned by `FinancialPPOTrainer`; TensorBoard does not
introduce a second reward or loss calculation.

For example, after the launcher prints the run directory, start the dashboard
with:

```bash
tensorboard --logdir /tmp/finance_train_logs/<run>/tensorboard \
  --host 0.0.0.0 --port 6006
```

The launcher prints the same command as `tensorboard_cmd`. Install the finance
extras in the selected interpreter before a real TensorBoard run:

```bash
/path/to/finance-venv/bin/python -m pip install -e '.[finance]'
```

The launcher follows the generic operational contract of the table-tennis
multi-GPU entrypoint. `NPROC_PER_NODE` is required to equal `NUM_GPUS`, and
`NUM_ENVS` is the per-rank count, so the effective environment batch is
`NUM_ENVS * NUM_GPUS`. The resolved `NCCL_*`, `TORCH_NCCL_ASYNC_ERROR_HANDLING`,
thread, device, rendezvous, and process-count settings are exported to every
`torchrun` worker and written to the launch log before the command starts.
Defaults are conservative for the shared host and can be overridden through
the environment when the machine has a known high-performance NCCL topology.

For a detached run, set `AUTO_TMUX=1` (and optionally `TMUX_ATTACH=1`). The
launcher writes a shell-quoted environment snapshot and wrapper under
`TMUX_ENV_FILE`/`TMUX_RUN_FILE`, appends process output to `TMUX_LOG_FILE`, and
keeps a failed tmux session open for inspection. The wrapper only re-enters
this finance launcher with `AUTO_TMUX=0`; it does not add rank isolation or
change archive, resume, reference/env sharding, gradient all-reduce, or
rank-0 checkpoint behavior. Isaac, JAX, Omniverse, curriculum, and isolated
launcher variables remain outside the finance contract.

Archive validation and sequence construction currently run independently on each
rank before the local slice is retained. The rollout/env ownership and gradient
work are sharded, but startup CPU/RAM is therefore replicated; use symbol/date
filters for bounded smoke runs when the full universe does not fit comfortably.

For a resume without an explicit `TRAJECTORY_ROOT` (or companion override), the
launcher omits source arguments and lets the checkpoint's saved canonical or
archive recipe decide. This keeps canonical checkpoints resumable through the
same command. `CURRICULUM_STAGES`, `JAX_RANK`, and related table-tennis-only
environment variables have no meaning for the finance task and are ignored.

Resume accepts both the native form and the table-tennis-style spelling:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 NUM_ENVS=1024 SEED=42 \
bash scripts/train_finance_sonic_4gpu.sh \
  --resume --checkpoint_path /path/to/checkpoint_040000.pt
```

On resume, architecture, sequence length, rollout length, epochs, and minibatch
count are inherited unless explicitly supplied. `--iterations` remains the
additional update count. Use `FINANCE_DRY_RUN=1` to inspect the fully resolved
command without opening data, a checkpoint, or a distributed process group.

Checkpoints contain actor/critic parameters, normalizers, optimizer state,
iteration, model/PPO/environment contracts, and a reference-pool recipe. The
recipe binds the canonical file's full SHA256/size, symbol/date selectors,
feature fields/units/source rules, observation-code fingerprint, reward
contract, and ordered sequence-index hash/count. It does not embed dataset
contents. Hashing the full file is intentional: market-context features also
depend on stocks outside the selected symbols.

Existing checkpoint paths are never overwritten. Resume with only
`--resume PATH_TO_CHECKPOINT --output-dir RUN_DIR --iterations N`; architecture,
PPO settings, normalization buffers, source and selectors are restored.
Explicitly conflicting settings are rejected. Resumption restarts reference
clips and clears KV history, not a bitwise continuation of a saved environment
step. Device, batch size and thread count remain runtime choices.

New training checkpoints use schema 4 and record the exact encoder denoising
contract, as well as the unchanged Reward V1 contract and reference recipe.
Schema 3 denotes earlier clean-input Reward V1 training; schema 2 introduced
reference provenance under the legacy reward. Training supports schemas 1-4.
Schemas 1/2 migrate from the legacy two-term reward to Reward V1 with encoder
denoising; schema 3 keeps Reward V1 and enables denoising. A shared loader first
validates the original schema, reward, reference and augmentation metadata, then
adapts only these known protocol changes. Actor, Critic, optimizer moments,
adaptive learning rate, iteration and normalization buffers are restored, not
reinitialized. Old Critic estimates may need adaptation when reward targets change.
The loader warns about migration; start logs, new schema-4 checkpoints and
evaluation reports retain `training_migrations` with source/target protocols and
the source iteration. Resuming the new checkpoint does not duplicate that record.
Unknown schemas, malformed source contracts, incompatible architecture/features/
data, or missing/mismatched schema-4 denoising metadata remain errors. This is
continued optimization under the current objective, not an exact replay of the
old objective or a weight-only warm start. Old checkpoints are never rewritten.

Each run also writes `run_config.json`, an immutable readable copy of the
initial recipe, including the denoising contract also printed in the start log.
Checkpoint metadata is authoritative; editing that JSON cannot
override a modern checkpoint. If the source file moves, supply
`--canonical NEW_PATH`; identical contents are accepted, changed contents are
not. This works when resuming into the same output directory: the initial JSON
keeps its original path, while new checkpoints record the resolved current path.
When migrating inside an old run directory, the original `run_config.json`
remains byte-for-byte unchanged. The CLI creates or validates
`run_config.schema4.json` as the active recipe, accepting only the recorded
reward/denoising changes and identical-file relocation. Further schema-4 resumes
in that directory use the same active recipe. Other configuration differences
still fail rather than overwriting either file.
Missing files, changed data, feature implementations or contracts fail closed.

### Checkpoint Reference Evaluation

Evaluate a saved model on its original reference pool, without specifying data
selectors again (choose a new report directory):

```bash
python -m scripts.eval_finance_sonic \
  --checkpoint /tmp/finance-ppo-example/checkpoint_000003.pt \
  --output-dir /tmp/finance-eval-example --device cpu --num-envs 16 --threads 2
```

The runner restores the Actor and fitted normalizers strictly, uses deterministic
mean actions, and visits each complete reference clip once in fixed order,
including a partial final batch. It clears observation history and Transformer
caches between independent batches and at exit. Evaluation does not optimize
parameters, refit statistics or update checkpoint files. Repeating evaluation
with different batch sizes should agree within floating-point tolerance. Schemas
1-4 remain evaluable: the report records `encoder_input_mode="clean"` and the
saved `encoder_denoising_contract` (null for pre-denoising schemas). Evaluation
does not randomly mask or corrupt the privileged reference.

Reports contain `summary.json`, `sequences.csv` and `symbols.csv`; existing
reports are never overwritten. The summary records both saved and resolved
reference recipes and coverage. Per-clip, per-symbol and overall metrics use
the same versioned tracking calculation as training. `monthly_mse`
measures normalized monthly log-return error; `cumulative_mse` measures its
cumulative path error divided by the square root of the horizon before
squaring. Reward V1 uses the four weighted kernels above and reports their
components and trajectory diagnostics. Legacy checkpoints are evaluated with
their original two-term reward, explicitly labeled by reward contract; scores
under different contracts are not directly comparable.
These metrics are not portfolio return, drawdown or unseen-data prediction
accuracy: actual future descriptors are still Encoder inputs.

### Sonic-Style Train-Set Playback Ledger

To inspect whether a checkpoint can trace the reference trajectories in the same
way as the L01 Sonic playback loop, request the optional detailed ledger:

```bash
python -m scripts.eval_finance_sonic \
  --checkpoint PATH_TO_CHECKPOINT \
  --output-dir /tmp/finance-playback-example \
  --dump-trajectories --max-sequences 8 --plot-sequences 4 --device cpu
```

`--max-sequences` is optional; omit it to visit the complete checkpoint reference
pool. `--dump-trajectories` requires `--output-dir` and streams rows directly to
disk instead of retaining the complete ledger in memory. The command writes the
usual aggregate files plus `trajectories.csv`.
There is one row for every `(sequence_id, symbol, anchor_period, target_period,
horizon)` pair. Prediction columns contain the normalized action mean and its
raw log-return conversion; target columns contain the corresponding reference
return, cumulative path, and the original future descriptor's cumulative value.
Rows are tagged with `rollout_mode=privileged_train_playback`,
`latent_source=oracle_future_encoder`, `encoder_input_mode=clean`, and
`action_mode=deterministic_mean`.

Each independent reference clip starts with an empty 32-step Decoder KV cache.
At the next anchor the runner supplies the actual current state and actual clean
future window from the same training clip; it never feeds the predicted path back
as a market state. Kin, denoising noise, Gaussian sampling, and portfolio/PnL
logic are not part of this diagnostic. Because the Encoder receives the true
future window, this report measures privileged low-level reference tracking only;
it is an upper-bound playback check, not a causal forecast or an unseen-data
backtest.

When a quick visual check is useful, add `--plot-sequences N` to the same
command. Dump mode then also writes `trajectory_overview.png`, a deterministic
1600x1000 RGB image containing three views: selected clips with predicted and
target cumulative paths, an absolute cumulative-error heatmap, and a
direction-agreement matrix based on monthly log-return signs. Only the first
`N` sequence IDs are rendered;
`trajectories.csv` still contains every evaluated row and remains the
authoritative source for metrics and further analysis. The PNG is generated
without Matplotlib, Pillow, or a display server, so it is suitable for batch
evaluation environments. Within the fixed canvas, the path panel shows up to
four clips and up to twelve representative anchors per clip. The heatmap and
direction matrix use those same clips and show at most 25 and 36 representative
anchors respectively.
The image headings and summary record displayed/total counts, while the CSV
retains every row. In `summary.json`, `plot_sequence_count`/`plot_anchor_count`
describe the subset sent to the image and the corresponding `*_total` fields
describe the complete evaluated playback. It is a diagnostic of privileged
train-set tracking, not a causal forecast, portfolio simulation, or PnL result.

`dataset_partition` is always `none`. Default evaluation is labeled
`training_reference_pool`, not a held-out validation/test set. To select another
pool explicitly, use, for example:

```bash
python -m scripts.eval_finance_sonic --checkpoint PATH_TO_CHECKPOINT \
  --custom-reference --symbols NVDA --start 2010-01 --end 2020-12 \
  --output-dir /tmp/finance-custom-eval-example
```

Unspecified selectors still inherit from the checkpoint. Custom evaluation is
labeled `custom_reference_pool` and retains checkpoint architecture,
normalization and feature/reward contracts; it does not guarantee disjoint
data. Changing symbols/dates without `--custom-reference` is rejected. The
identical-file relocation exception remains available in default evaluation.

Checkpoints without reference metadata require an explicit
`--reference-config PATH_TO_REFERENCE_JSON` for evaluation. That JSON must contain
a valid recipe with the checkpoint's matching reward contract, directly or inside
a `run_config.json` wrapper. It supplies the intended pool but cannot prove what
the old model trained on. Reports retain
`legacy_reference_unverified`; even a later custom evaluation keeps that flag.
The training CLI also requires this original recipe for any supported checkpoint
without reference metadata, including schema 1 and checkpoints saved through the
direct trainer API. The recipe must match the original reward (legacy for schemas
1/2, V1 for schemas 3/4); it is validated before migration, and the new checkpoint
retains `legacy_reference_unverified`. A saved reference cannot be overridden.
Direct trainer callers may continue without a historical recipe, but the saved
lineage is also marked unverified even if the trainer has a current pool recipe.
The separate `check_finance_sonic` command below remains an architecture smoke
check, not this frozen-checkpoint evaluator.

The following standalone example is retained only for supervised diagnostics,
not as the Sonic-style PPO training entry point:

```python
from dataclasses import asdict
import torch
from gear_sonic.finance.model import FinancialSonic, FinancialSonicConfig
from gear_sonic.finance.losses import financial_sonic_loss

model = FinancialSonic(FinancialSonicConfig())
# train_current: [B,S,16]; train_future: [B,S,10,15]
model.fit_normalizers(train_current, train_future)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.zero_grad()
out = model(train_current, train_future)
losses = financial_sonic_loss(out, train_future,
                             return_scale=model.future_normalizer.scale[0])
losses["total"].backward()
optimizer.step()
torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, checkpoint_path)

model.eval()
model.reset_cache()
# current_month: [B,16]; external_raw_latent: [B,64]
forecast = model.predict_step(current_month, external_raw_latent)
```

Run `python -m scripts.check_finance_sonic --canonical PATH_TO_MONTHLY_CANONICAL
--symbols AAPL MSFT` from the repository root (on one shell line). This reads
all-universe market context, fits statistics on all selected-stock sequences,
checks a backward pass without changing weights, and compares full/streaming
conditional reconstruction on the same reference pool. Optional date arguments
can narrow either operation. `--tiny` selects a small CPU model;
default dimensions are the full architecture above. This command is a bounded
pipeline check, not a training run or backtest.

Run `python -m pytest tests/test_finance_observations.py tests/test_finance_model.py`.

## Feature Outlier Preparation

`python -m scripts.prepare_finance_features --canonical PATH --output-dir NEW_DIR
--as-of 2026-09-08` exports the current 16-state/10x15-reference schema. It uses
completed months only, constructs market context from the complete available
universe, and drops windows crossing missing months. Six rolling-feature warm-up
months are required before the anchor. It exports every eligible anchor, not
only the 32-anchor blocks returned by `iter_sequences`.

The output is one full reference pool with no quality weighting. Each current
anchor contributes once to current normalization statistics; all ten reference
frames contribute to future statistics. Overlapping future frames therefore
occur multiple times, and anchor-relative features are intentionally distinct.
`symbol_coverage.csv` reports the probability of each symbol under uniform-window
sampling; it does not alter sampling weights. Short windows do not themselves
equalize stock coverage. Sonic's robot motion sampler has separate length-aware
and failure-adaptive sampling; such a sampler is deferred here.

Files in `data/us_socket/月线特征处理/sonic_reference_v1`:

- `current_raw.npy`: `[N,16]` current observations in feature units.
- `future_raw.npy`: `[N,10,15]` references in feature units, including original
  monthly log returns for dynamic tracking targets.
- `current_normalized.npy`, `future_normalized.npy`: median/IQR inputs bounded
  to [-10,10]. This controls numerical magnitude, not market-data authenticity.
- `current_clipped.npy`, `future_clipped.npy`: per-cell clipping indicators,
  not validity masks. Exported windows contain no missing values.
- `normalization_stats.json`, `feature_outlier_report.json`: fixed feature
  statistics, diagnostic quantiles, thresholds and clipping counts.
- `sample_index.csv`, `symbol_coverage.csv`, `feature_schema.json`,
  `manifest.json`: identities, coverage, field order, source hash and timing.

Keep data in raw feature units when passing it through `FinancialSonic`; the
model loads the exported statistics and normalizes internally. The normalized
arrays are provided for direct bounded-input consumers and inspection. Passing
them through the model normalizers again would normalize twice. The dynamic
return target remains raw; kin reconstructs bounded descriptors as before.

```python
model.load_normalizers("data/us_socket/月线特征处理/sonic_reference_v1/normalization_stats.json")
# current_raw: [B,S,16], future_raw: [B,S,10,15]
output = model(current_raw, future_raw)
```

## Variable-Length Monthly Trajectories

The cleaned canonical table is also organized as a Sonic-style trajectory
archive by running:

```bash
python -m scripts.reconstruct_monthly_trajectories \
  --canonical data/us_socket/月线清洗重构/monthly_canonical.csv \
  --event-breaks data/us_socket/月线清洗重构/monthly_event_breaks.csv \
  --output-dir data/us_socket/月线轨迹重构
```

`data/us_socket/月线轨迹重构/trajectories/` contains one mapping-wrapped
`<safe_symbol>__segment_<id>.pkl` for each continuous valid monthly run. A
segment is closed by an invalid raw/QFQ bar, a missing month, or an enabled
confirmed event boundary. Short segments are retained for provenance but have
zero complete six-month-warmup plus ten-month-horizon anchors. The PKL stores
raw, forward-adjusted, and optional backward-adjusted OHLC fields, raw volume /
amount / turnover, validity masks, dates, quality flags, and source row numbers.
It deliberately does not store normalized 16-dimensional observations or a
fake FPS field; those are derived by a finance loader so market context and
normalization statistics remain current.

`metadata.pkl`, `trajectory_index.csv`, `excluded_rows.csv`, and `manifest.json`
record lengths, anchor counts, exclusions, event decisions, schema/dtype, and
the canonical SHA256. Training can now use this archive directly with
`--trajectory-root`; all companion files and every indexed PKL are checked before
training. Unless `--canonical` is also supplied as an explicit market-context
override, context is reconstructed from the archive itself. The resolved archive
fingerprints and ordered fixed-sequence identity are stored in every checkpoint.

Legacy support: `python -m scripts.prepare_legacy_features --monthly-features
PATH --playback-samples PATH --output-dir NEW_DIR` bounds the earlier 6x16/10x16
format, including reported turnover. It fits on unique symbol-month rows (not
repeated overlapping windows), preserves the `data_valid` bit, and writes
distinct missing and clipping masks. Its output under
`data/us_socket/月线特征处理/legacy_playback_v1` retains the 318,888 original
sample identities and order. Missing monthly values use zero placeholders plus
`monthly_missing.npy`; these are not imputed observations. Original JSONL is
still the source for raw legacy trajectories. The legacy format and its
normalization file cannot be passed to the current 16/15 model interface.

Full preparation on 2026-09-08 retained all 318,888 legacy windows (56.43 seconds);
71,415 windows contain at least one clipped feature. Reported turnover was bounded
in 13,871 unique monthly rows. The current schema exported 848,518 anchors from
9,978 symbols (406.90 seconds), with 29,386 clipped current-feature cells and
292,673 clipped future-descriptor cells. These are input clipping counts, not
verified erroneous prices or deleted trajectories. Every exported normalized
array was checked for finite values and bounds. A real 32-anchor CPU model check
loaded the exported statistics, matched NumPy normalization, produced finite
nonzero encoder gradients, and matched streaming/full output within 1e-5.

## Encoder/Decoder Verification on 2026-09-08

The earlier encoder/decoder implementation passed 28 tests on CPU. Coverage includes
feature causality, gap/split boundaries, shared-latent routing, each loss branch's
encoder gradients, mask handling, normalizer checkpoint round-trips, burn-in,
causal attention, streaming equivalence beyond 32 steps, and partial batch resets.
The full 11,085,792-parameter configuration also passed the real canonical-data
check for AAPL/MSFT: current `[2,32,16]`, future `[2,32,10,15]`, dyn `[2,32,10]`,
and maximum streaming/full difference `1.0058284e-7`.

This workspace used the existing Python/PyTorch interpreter at
`/shared_disk/users/wuyutong/AeroStep_Sonic/.venv_sim/bin/python` with added FSQ/test
packages in `/tmp/finance-sonic-deps` via `PYTHONPATH`. The reference environment
was not modified. A CUDA initialization warning appeared during CPU backward;
GPU execution was not tested. These results verify implementation behavior,
not learned forecasting accuracy.

## PPO Verification on 2026-09-08

After adding the financial Actor/Critic adapters, playback environment, PPO
runner and CLI, the complete available repository test suite passed: 113 tests
and three subtests on CPU (`python -m pytest tests -q`). Independent code reviews
found no high- or medium-severity issues. The original Transformer decoder file
still has the same SHA-256 as the reference implementation.

Bounded checks on the real AAPL/MSFT canonical data used four environment streams
and four reference clips of 64 anchors each, without an automatic date split:

- Tiny network: three 24-step PPO iterations, exercising more than 32 cache steps
  and four terminal resets. Reloading iteration 3 completed iteration 4.
- Original network dimensions: two iterations with the default five epochs and
  four minibatches, producing 40 optimizer updates in total.
- Both configurations produced finite training metrics and saved checkpoints.
  Checkpoint overwrite refusal and configuration validation are covered by tests.

Smoke artifacts are under `/tmp/finance-sonic-ppo-smoke.QO5AIE`:
`full/checkpoint_000002.pt`, `tiny/checkpoint_000003.pt`, and
`tiny_resume/checkpoint_000004.pt`. These are temporary verification artifacts,
not trained forecasting models.

The full-network smoke run had large iteration-average KL values (about 651.8
and 12.94 versus target 0.01); adaptive learning rate reached its reference floor
of `1e-5`. This is a stability warning, not evidence of convergence. No defaults
were tuned to these two stocks. A separate original-size no-update replay check,
with and without cached prefixes, matched rollout means within `4.92e-7`;
pre-update KL was `0.00010014`, from the retained formula's epsilon. This does
not indicate an initial rollout/replay mismatch. The KL target adjusts learning
rate but is not a hard update-stopping threshold. GPU/distributed execution and
long-run learning quality remain unverified; all training jobs in this check
were bounded CPU runs.

## Checkpoint Inheritance Verification on 2026-09-08

After the reference metadata and frozen evaluator changes, the full repository
suite passed 148 tests and three subtests on CPU. Separate specification and
code-quality reviews found no outstanding substantive issues. The local test
command was:

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/tmp/finance-sonic-deps \
  /shared_disk/users/wuyutong/AeroStep_Sonic/.venv_sim/bin/python -m pytest tests -q
```

A bounded real AAPL/MSFT check selected 2010-01 through 2020-12 and produced two
complete 64-anchor clips. One tiny PPO iteration saved a metadata-bound
checkpoint. Default evaluation visited all 128 anchors without supplying data
selectors. Repeating with the same batch size produced identical metrics;
changing batch size from three to one changed overall metrics by at most
`1.44e-9`. Evaluation left checkpoint bytes unchanged. Resuming with only the
checkpoint, output directory, iteration count and runtime batch size reached
iteration two, preserving reference metadata and all six normalization buffers.
The canonical file SHA256 remained
`85e8027af56088091a2b983fd43c41901aab0876be3715a574466eafebc22c01`.

Temporary artifacts are under `/tmp/finance-inheritance-smoke.U0IS0q`:
`train/checkpoint_000001.pt`, `train/checkpoint_000002.pt`,
`train/run_config.json`, and reports in `eval_first` and `eval_repeat`.
This verifies inheritance and replay mechanics, not training convergence.
Full-pool memory/time and CUDA reproducibility remain unmeasured.

## Reward V1 Verification on 2026-09-09

The four-term reward is active in training and frozen evaluation. Test-driven
implementation added independent formula, precision, rolling-boundary,
normalization and versioned-checkpoint checks. The complete repository suite
passed 222 tests and three subtests on CPU. Separate specification and
code-quality reviews found no outstanding actionable issues.

Bounded checks used the unchanged canonical AAPL/MSFT pool: four 64-anchor clips,
four training environments, no automatic data partition, and two CPU threads.

- Tiny model: three 24-step iterations crossed the 32-step cache window and
  triggered four terminal resets. Rolling valid counts were 92, 96 and 92 out
  of 96 transitions per iteration. All four reward components and losses were
  finite. Resume in the same output directory reached iteration four, restarted
  histories, and preserved the reference identity and all normalization buffers.
- Original model: two 24-step iterations with the unchanged five epochs and
  four minibatches produced 40 optimizer updates. Losses, reward diagnostics,
  model parameters and optimizer tensors were finite. The saved schema-3
  checkpoint retained the original network widths and only kin/cycle auxiliary
  losses. This full-size run crossed the cache window but not a clip terminal.
- V1 frozen evaluation visited all 256 anchors once, with 252 valid rolling
  comparisons. Batch sizes three and one agreed within floating-point tolerance;
  checkpoint bytes were unchanged. The report explicitly uses deterministic
  mean actions, unlike sampled-action training rewards.
- A real schema-2 checkpoint still evaluated all 128 anchors of its original
  pool with the legacy 0.5/0.5 reward. Attempting V1 training resume rejected it
  before creating output files. Legacy and V1 scores are not interchangeable.

Temporary verification artifacts are under `/tmp/finance-reward-v1-G7Nbz7`:
`tiny/checkpoint_000003.pt`, `tiny/checkpoint_000004.pt`,
`full/checkpoint_000002.pt`, training JSONL logs, and `eval`/`legacy_eval` reports.
The canonical SHA256 above and the reference-matching Transformer SHA256 remained
unchanged. No reference repository, source data or older artifact was modified.

The full-model iteration-average KL was approximately 649.40 and 23.12, versus
the unchanged target of 0.01; the adaptive learning rate reached `1e-5`.
This remains an optimization-stability warning. The bounded checks establish
pipeline correctness, not convergence, forecasting accuracy or investment
performance. CUDA initialization still emitted the existing warning during CPU
backward; GPU and distributed execution were not tested.

## Kin Ordering and Denoising Verification on 2026-09-09

After the changes described in Encoder Denoising above, the complete financial
CPU suite passed 269 tests and three subtests. This includes stored-corruption
PPO mean/log-probability parity across the 32-step window and episode resets,
plus explicit observation-history cache rebuilding with identical KV tensors
and unchanged RNG state. Separate specification and code-quality reviews found
no outstanding actionable issues.

The robot repository changed only G1AnchorReconLoss's flat target unpacking and
its reconstruction regression tests. An actual-source AST CPU harness loads the
real loss/helpers, UniversalTokenModule, observation builder/accessors, and
repository test definitions without simulator import side effects. Before the
fix it reported 13 failures and eight passes; after the fix all 21 cases passed.
Semantically exact reconstruction now has zero loss, including the actual
clean-target builder case. Full Hydra/IsaacLab integration was not exercised.

A bounded real-data smoke used AAPL/MSFT anchors from 2015-01 through 2020-12,
14 eight-anchor clips, four environments, the tiny network, 12 rollout steps,
two epochs, and two minibatches. Two iterations followed by one resumed iteration
completed with finite losses. All 205 saved model/optimizer tensors checked were
finite. Schema4 and the exact denoising contract survived resume; clean frozen
evaluation visited all 112 anchors. These checks establish pipeline behavior,
not convergence, deployable prediction quality, or PnL.

Artifacts: `/tmp/finance-denoising-smoke-K08gig/train/checkpoint_000003.pt`,
`train/run_config.json`, and `eval/summary.json`, `eval/sequences.csv`,
`eval/symbols.csv` under the same temporary root. The robot CPU harness is
`/tmp/test_stage5_cpu_harness.py`. Canonical data, feature implementation and
Transformer hashes stayed unchanged. Both repository diff checks passed.
The existing CUDA initialization warning occurred during CPU backward; GPU
execution/performance and distributed training remain unverified.

## Legacy Training Resume Verification on 2026-09-09

The later user request replaces the historical legacy-resume rejection policy
with the full-state migration described above. The complete financial CPU suite
passed 412 tests and three subtests. Regression tests compare restored Actor,
Critic, optimizer, LR, iteration and normalizer values exactly for schemas 1/2/3,
then execute another PPO iteration. They also cover repeated same-directory
resume with immutable original recipes, strict source-protocol rejection,
schema-1 CLI recovery with an explicit original recipe, and unverified provenance
for direct-API resumes without a historical recipe.

Actual historical tiny checkpoints also resumed against the canonical data:
schema 2 continued iteration 2 to 3, and schema 3 continued iteration 4 to 5.
Each resulting checkpoint contained 205 finite model/optimizer tensors. Clean
frozen evaluation covered all 128 and 256 original-pool anchors respectively;
reports preserved the migration histories and current contracts. SHA256 checks
confirmed both source checkpoints and their original run configurations were
unchanged. These bounded checks establish resume behavior, not convergence or
forecasting performance. GPU and distributed behavior were not exercised; the
existing CUDA initialization warning occurred during CPU backward.

Artifacts are under `/tmp/finance-legacy-resume-1U7nJu`: migrated checkpoints
`schema2/checkpoint_000003.pt` and `schema3/checkpoint_000005.pt`, with reports in
`schema2_eval` and `schema3_eval`. The implementation plan and current policy are
in `superpowers/plans/2026-09-09-legacy-training-resume.md`.

## Distributed Archive Verification on 2026-09-15

A two-rank CPU/gloo `torch.distributed.run` test executes a complete PPO update
against a synthetic trajectory archive. Its 44 fixed clips split 22/22, with
four environments on each rank and eight globally. Both ranks report the same
post-update parameter checksum. The output contains one `run_config.json` and
one rank-0 checkpoint; the checkpoint records `gradient_reduction=all_reduce`,
rank/world/reference/environment counts, finite model and optimizer state, and a
Critic running-stat count pooled across both ranks.

The production archive at `data/us_socket/月线轨迹重构` was also checked directly.
An AAPL tiny-model run loaded and validated all 23,326 indexed archive files,
built 86 two-anchor clips, completed one PPO update, and resumed from its sole
checkpoint for a second update without canonical/source arguments. These are
pipeline checks, not convergence or return evidence. The current host did not
provide usable CUDA, so NCCL/GPU performance remains to be measured on the
training machine.
