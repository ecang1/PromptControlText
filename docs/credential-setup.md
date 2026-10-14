# Credentials for the frontier-model extension

This setup supports one generator from OpenAI, Anthropic, and Google, plus the
existing `meta-llama/Llama-3.1-8B-Instruct` judge. It does not launch inference,
create paid resources, or make the current vLLM runner support provider APIs.
Provider adapters and benchmark-context fixes are separate work.

## Where to enter keys

Keep the actual keys in `.env` in the repository root on your computer. The file
is ignored by Git; `.env.example` is a blank template safe to publish. Do not
paste keys into chat, commit messages, issues, PRs, or source code. GitHub Actions
secrets are only needed if an Actions workflow is later introduced; they do not
make credentials available to this local session.

For setup alone, install only the small existing dependency, not vLLM:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install python-dotenv
.venv/bin/python scripts/credentials.py init
```

Open `.env` in your editor and paste each key after its equals sign. Start with
`RUNPOD_API_KEY`; the other entries may remain blank until their accounts are ready.
`init` creates the file with mode 600 and never overwrites it. This is a local
plaintext file restricted to your user, not an encrypted vault. If an editor
changes its permissions, run `chmod 600 .env`.

| Variable | Where to create it | Access needed |
| --- | --- | --- |
| `RUNPOD_API_KEY` | [Runpod Credentials](https://docs.runpod.io/get-started/credentials), API Keys tab | Account reads and pod management for the planned worker; select the relevant restricted permissions offered by the console. A read-only key cannot provision pods. |
| `OPENAI_API_KEY` | [OpenAI API setup](https://developers.openai.com/api/docs/quickstart) | A project with API billing and access to the selected generator; Batch permissions if used. |
| `ANTHROPIC_API_KEY` | [Claude API setup](https://platform.claude.com/docs/en/api/overview) | A workspace with API billing and access to the selected generator. |
| `ANTHROPIC_WORKSPACE_ID` (optional) | [Claude workspace settings](https://platform.claude.com/docs/en/manage-claude/workspaces) | Required when the API key is not scoped to one workspace; the runner sends it as the `anthropic-workspace-id` header. |
| `GEMINI_API_KEY` | [Google AI Studio key setup](https://ai.google.dev/gemini-api/docs/api-key) | A project with the required billing, quota, and model access. Use this variable consistently rather than also setting `GOOGLE_API_KEY`. |
| `HF_TOKEN` | [Hugging Face tokens](https://huggingface.co/settings/tokens) | Read access and approved access to [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct). Both checkpoints are approved for this account. |

A Runpod website login provides console access, but does not authenticate local
CLI calls. The Runpod API key enables those calls. A plugin connection, if
available, is a separate connection; installing or mentioning a plugin is not
proof of account access. The connected GitHub plugin can record the setup without
adding a GitHub token to `.env`.

## Check presence without printing secrets

```bash
.venv/bin/python scripts/credentials.py status
git check-ignore .env
```

`status` reports only presence, makes no network requests, and returns 1 while
any required credential is missing. The workspace ID is optional for scoped keys.
SET does not establish a valid key, balance, permissions,
or model access. The helper accepts a global `--env-file PATH` option if needed.
Nonempty existing environment variables take precedence over `.env`.

To pass keys to a command without putting them in shell history or expanding
shell code from the file:

```bash
.venv/bin/python scripts/credentials.py run -- runpodctl user
```

Do not source `.env` or use commands that print the environment. The helper
passes only the five documented credential names and optional workspace ID from the file; a child process
can still print its own output, so use tools with appropriate logging.

If Anthropic returns an error saying the key is not scoped to a workspace, add
`ANTHROPIC_WORKSPACE_ID=wrkspc_your_workspace_id` on a separate line in the same
local `.env` file. Use an actual workspace ID from the organization that owns the
key, not the workspace name or organization ID. Alternatively, use a key scoped
to that workspace. This changes account routing, not prompts or model settings.
Do not retry until the configuration has been updated.

## Runpod CLI and remote execution

Install the CLI using the [official instructions](https://docs.runpod.io/runpodctl/overview)
or a binary from [official releases](https://github.com/runpod/runpodctl/releases).
A project-local executable can live at `.local/bin/runpodctl`, which Git ignores:

```bash
.venv/bin/python scripts/credentials.py run -- .local/bin/runpodctl user
```

Once the key is entered, check the account and list existing resources before
creating anything. Use the CLI's live help for current commands and permissions.
Configure SSH before provisioning a worker. API-provider generation runs through
provider APIs; a GPU is needed for the Llama judge, not for API submission.

For remote work, give each worker only the credentials it needs. Runpod's
[Secrets facility](https://docs.runpod.io/get-started/credentials) supports
injecting values into pods. Its documented secret names use `RUNPOD_SECRET_`;
map those names to the SDK's expected variables inside the worker without
printing values. Do not copy the entire local `.env` to every worker or bake it
into an image. Keep the Runpod management key on the orchestration machine.

## What belongs in the GitHub record

Commit the template, setup instructions, helper, and tests. Record credential
presence and verification outcomes without key values or account details.
Credentials, local executables, and virtual environments remain untracked.
Authentication is established only after the user enters keys and live checks
succeed. See the [verification record](credential-verification.md) for the
checks completed on this setup and their limits.
The research spending ceiling remains $250; this setup performs no paid run.
