# Composition study serving runbook

This runbook turns the frozen settings in
`artifacts/composition_v1_generation_config.json` into a per-checkpoint vLLM
serving procedure on RunPod Pods. It does not provision compute or request
Hugging Face access. Use the same frozen item manifest for every generator
checkpoint.

## Current readiness

| Checkpoint | Hugging Face access | Serving profile |
| --- | --- | --- |
| `meta-llama/Llama-3.1-8B-Instruct` | Approved | BF16, tensor parallel 1 |
| `meta-llama/Llama-3.3-70B-Instruct` | Approved | BF16, tensor parallel 4 |
| `Qwen/Qwen2-7B-Instruct` | Public checkpoint | BF16, tensor parallel 1 |
| `Qwen/Qwen2.5-72B-Instruct` | Public checkpoint | BF16, tensor parallel 4 |

The configured reference is vLLM 0.19.0 on A100 80GB SXM GPUs. The 70B/72B
profiles therefore require four GPUs for a paper-matched run; the 7B/8B
profiles use one. If the selected host differs, record the hardware/runtime
deviation before confirmatory generation. Do not change the frozen generation
settings to make a run fit.

## Host and access prerequisites

1. Create a RunPod **Pod** for the experiment rather than a serverless endpoint:
   the runner and the vLLM server can run together in the Pod and use the
   private localhost endpoint. Use one Pod with one GPU for each 7B/8B run and
   one Pod with four GPUs for each 70B/72B run. Select A100 80GB SXM when
   available to match the frozen reference. Keep the four tensor-parallel GPUs
   in the same Pod; do not split them across separate Pods.
2. Attach a persistent volume when creating the Pod. A RunPod network volume
   survives Pod deletion and can move between Pods, but is pinned to its data
   center and replaces the Pod's `/workspace` volume disk. Put the repository,
   model files, manifest, and output CSVs under `/workspace`. The four BF16
   checkpoints together need roughly 300+ GB of storage before cache/output
   headroom; provision about 500 GB if keeping all four. If reusing one smaller
   volume and replacing model files between runs, leave ample headroom for the
   largest 70B/72B model download and cache. Choose the RunPod data center when
   creating the volume, then select a Pod in that same location.
3. Check the live Pod selection for the number/type of GPUs, available
   capacity, and total hourly cost before deployment; those vary by location
   and cloud tier. The frozen four-GPU A100 SXM reference can cost materially
   more than the single-GPU profile while the Pod is running.
4. Install the repository environment and vLLM 0.19.0 in the Pod. Confirm
   CUDA/NCCL and the installed vLLM build match the GPU environment.
5. On the Pod that downloads gated weights, authenticate with the user's own
   Hugging Face read token using `hf auth login` (or inject `HF_TOKEN` from the
   host's secret manager). Never put the token in this repository, a command
   committed to shell history, or a chat message.
6. Download either approved gated checkpoint using the user's Hugging Face
   token. Both Llama 3.1 8B and Llama 3.3 70B are approved for this account.
   Public Qwen checkpoints do not need the gated-model token.

For an interactive authenticated download, the Hugging Face CLI command is:

```bash
MODEL_DIR=/workspace/models
mkdir -p "$MODEL_DIR"
hf download meta-llama/Llama-3.3-70B-Instruct \
  --local-dir "$MODEL_DIR/meta-llama--Llama-3.3-70B-Instruct"
```

Keep `/workspace` on the Pod's attached persistent volume. Ensure the token is
available to the CLI from its protected login store or secret injection. Do not
use `--local-dir` inside the source checkout.

## Start and verify one vLLM server

Run this inside the activated vLLM environment on the GPU host. Replace
`REPO_ID`, `MODEL_PATH`, and `TP` using the row for the checkpoint. Keep the
served model name equal to the exact repo ID: the runner sends that exact name
in each request.

```bash
REPO_ID=meta-llama/Llama-3.3-70B-Instruct
MODEL_PATH="$MODEL_DIR/meta-llama--Llama-3.3-70B-Instruct"
TP=4

vllm serve "$MODEL_PATH" \
  --served-model-name "$REPO_ID" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size "$TP" \
  --max-num-seqs 512
```

The local (same-Pod) endpoint is `http://localhost:8000/v1`; keep generation in
the same Pod so the vLLM port does not need public exposure. If you later run
generation elsewhere, use a private, access-controlled tunnel/address and
configure authentication before passing a remote `--base-url`; do not expose an
unauthenticated vLLM port to the public internet. vLLM applies the checkpoint
tokenizer's chat template by default. Avoid extra flags that change sampling or
quantization.

In a second terminal, check readiness and that the exact served model name is
registered:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models
```

Do not start the confirmatory run unless health succeeds and `/v1/models`
contains the exact `REPO_ID` value.

## Run generation

From the repository checkout under `$HOME` on the Pod, with the server
reachable at the configured endpoint. Keep the output CSV on the persistent
`/workspace` volume:

```bash
python -m experiments.run_composition \
  --generator-model meta-llama/Llama-3.3-70B-Instruct \
  --base-url http://localhost:8000/v1 \
  --manifest artifacts/composition_v1_confirmatory_manifest.json \
  --output-file /workspace/results/composition_v1_confirmatory_llama33_70b.csv \
  --max-workers 512
```

Run once per checkpoint, using a distinct output file for each. Use
`--resume` with the same model, endpoint, manifest, and output file after an
interruption. The runner enforces the frozen sampling settings and the
initial-plus-three retry policy. Failed/exhausted rows remain in the CSV with
their attempt count and error; do not remove them before reporting.

Before launching, verify the manifest hash matches the design freeze record,
record host/GPU model, vLLM version, endpoint, model path/revision, and run
start time in the run log. Keep credentials out of that log. This runbook is
operational documentation and does not alter the frozen design or its hashes.

## RunPod references

- [Choose a Pod](https://docs.runpod.io/pods/choose-a-pod)
- [Pod storage options](https://docs.runpod.io/pods/storage/types)
- [Network volumes](https://docs.runpod.io/storage/network-volumes)
- [GPU Pod pricing](https://www.runpod.io/pricing)
