# RunPod Pod setup guide for the composition experiment

This guide prepares one RunPod GPU Pod to host vLLM and run the composition
generation runner. It stops before creating the Pod, which starts compute
billing. The serving profile follows
`artifacts/composition_v1_generation_config.json`.

## Target configuration

| Setting | Select |
| --- | --- |
| RunPod product | GPU Pod (not Serverless) |
| Cloud tier | Secure Cloud is the recommended default; compare current availability and price in the console |
| GPU | NVIDIA A100 SXM, 80 GB, when available |
| GPU count | 4 for Llama 3.3 70B and Qwen 2.5 72B; 1 for Llama 3.1 8B and Qwen 2 7B |
| Persistent storage | 500 GB Network Volume, mounted at `/workspace` |
| Container image | Official RunPod PyTorch template with a CUDA version compatible with the vLLM 0.19.0 build |
| vLLM endpoint | Local to the Pod at `http://localhost:8000/v1`; do not expose port 8000 publicly |

Keep all four tensor-parallel GPUs in the same Pod. The two 70B-class models
need about 140 GB of GPU memory for BF16 weights, so the frozen four-GPU
profile is required. A100 SXM 80 GB is the reference; selecting another GPU
model is a documented hardware deviation and must be recorded before
confirmatory generation.

## Before opening the deploy form

1. Sign in to RunPod, add billing information, and enable two-factor
   authentication.
2. Register an SSH public key in your RunPod account if you want SSH access.
   You can also use the Pod's web terminal/Jupyter access if enabled by the
   selected official template.
3. Decide whether to keep all four model checkpoints on persistent storage.
   Their BF16 weights together are roughly 300+ GB before cache, datasets, and
   results; use about 500 GB to leave working room. A RunPod Network Volume is
   a separate monthly storage charge and is tied to one data center.

## Create persistent storage first

1. In the RunPod console, open **Storage** and choose **New Network Volume**.
2. Name it `promptcontrol-composition-v1`.
3. Choose **Standard** storage and set capacity to **500 GB**.
4. Select a data center. Remember this selection; the Pod must use the same
   data center to attach this volume.
5. Create the volume. Do not put Hugging Face tokens in the volume name,
   metadata, or any file committed to the repository.

Network volumes survive Pod deletion and can be attached to a replacement Pod.
When attached, the volume mounts at `/workspace` and replaces the Pod's normal
volume disk. Keep model weights and output CSVs there. Clone the Git repository
under `$HOME` on the Pod's container filesystem instead: Git needs to set POSIX
permissions on files under `.git`, and a mounted volume may reject that
operation. Container-disk code is temporary, so record the Git commit hash and
reclone that exact revision after a Pod restart if needed. Back up final CSVs
separately before deleting the Network Volume.

## Configure the GPU Pod

1. From the RunPod console, select **+ New → Pod** (or **Pods → Deploy**).
2. Choose an **official RunPod PyTorch template**. Use a template with SSH
   access enabled if you want command-line access. Avoid templates that
   automatically download a different model or start a model server on an
   unknown port.
3. Name the Pod `promptcontrol-composition-v1`.
4. Select the same data center as the Network Volume.
5. Search for **NVIDIA A100 SXM 80GB** and select it if available. Select
   **GPU count: 4** for the full experiment's server profile. This is the
   target profile for running the two 70B-class models. The smaller models
   need one GPU, but the 4-GPU Pod can serve them sequentially too.
6. Attach the Network Volume `promptcontrol-composition-v1` and confirm its
   mount path is `/workspace`.
7. Set the container disk to **50 GB** for the operating system and temporary
   files. Keep model weights, repo, manifests, and results on `/workspace`.
8. Enable SSH access if desired; add your registered public key if prompted.
   Jupyter is optional. Do **not** add a public HTTP mapping for port `8000`;
   vLLM and the generation runner will run inside the same Pod.
9. Leave startup/override commands blank for now. We will start vLLM manually
   after installing the pinned environment and checking the GPU/volume.
10. Review the pricing summary, especially GPU count, hourly total, persistent
    storage, and stopped-Pod charges. Check current availability and total cost
    in the console, then select **Deploy On-Demand** when ready. RunPod bills
    GPU Pods while running, and storage can continue to incur charges while
    the Pod is stopped.

## First login checks

After the Pod reports **Running**, open its terminal or connect over SSH and
check the hardware and persistent mount:

```bash
nvidia-smi
df -h /workspace
ls -la /workspace
```

Confirm four A100 SXM 80GB devices are visible for the full target setup and
that `/workspace` is the attached Network Volume. If the GPU count/type or
mount is wrong, stop before downloading models or starting vLLM.

## Put the experiment files on the Pod

Use the exact repository revision that contains the frozen composition runner
and confirmatory manifest. Clone it under the Pod user's home directory, not
under `/workspace`; this avoids Git's `.git/config.lock` permission failure on
some mounted volumes. If those changes have not been pushed to GitHub, push the
branch first or transfer the checkout into `$HOME` without cloning on the
mounted volume.

Use this layout:

```text
/root/PromptControlText/            # Git checkout (or use "$HOME/PromptControlText")
/workspace/models/                  # checkpoint weights
/workspace/results/                 # generated CSVs and run logs
```

The datasets and frozen manifest are included in the repository checkout and
can be read from `$HOME/PromptControlText`. The runner can write results to
`/workspace/results` using an absolute `--output-file` path.

Install vLLM **0.19.0** and the repository's Python requirements in the Pod's
Linux environment. Both `meta-llama/Llama-3.1-8B-Instruct` and
`meta-llama/Llama-3.3-70B-Instruct` are approved for this Hugging Face account.
Keep Hugging Face authentication in the user's protected CLI login or a secret
manager; do not save the token in the repo, a notebook, or a shell command
recorded in history. The two public Qwen checkpoints do not require this token.

## Stop and preserve data

- Stop the Pod when work is paused to release GPU compute. Persistent network
  volume storage remains billable while the Pod is stopped.
- Before terminating a Pod, confirm code, model weights, and output CSVs are on
  the attached Network Volume or copied to a separate backup. Data on the
  container disk is temporary; Pod volume-disk data is deleted when the Pod is
  terminated.
- Retain the Network Volume until the confirmatory result files and logs have
  been copied and verified elsewhere.

## References

- [RunPod: Deploy your first Pod](https://docs.runpod.io/get-started)
- [RunPod: Choose a Pod](https://docs.runpod.io/pods/choose-a-pod)
- [RunPod: Storage options](https://docs.runpod.io/pods/storage/types)
- [RunPod: Network volumes](https://docs.runpod.io/storage/network-volumes)
- [RunPod: Manage Pods](https://docs.runpod.io/pods/manage-pods)
- [RunPod: current GPU pricing](https://www.runpod.io/pricing)
- [vLLM: OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)
