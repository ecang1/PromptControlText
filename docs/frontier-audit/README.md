# Human-validation preparation

Start with [ANNOTATOR_GUIDE.md](ANNOTATOR_GUIDE.md) for the criteria and [COORDINATOR.md](COORDINATOR.md) for the full workflow and commands. [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) is intentionally unfilled; [VALIDATION.md](VALIDATION.md) records software checks.

The separate `audit.frontier` dashboard uses the six canonical policies, response-only case display, independent insert-only first passes, separate sealed-source adjudication, confidence, evidence and a review flag. Historical audit files and the canonical judge remain unchanged.

A provisional 30-response practice packet was prepared locally from the verified partial generation archive. It contains **no human ratings**. Coordinator-private runtime artifacts belong under `.local/frontier-audit/`; they are not published with this code. The prepared shareable folder/archive is `practice-kit-v1` / `practice-kit-v1.zip` in that directory. Give each annotator their own copy, choose pseudonyms, and run inside each kit:

```sh
python3 -m audit.frontier.dashboard --bundle bundle.json \
  --database .local/ratings.sqlite --coder rater_a --port 8765
```

Use `rater_b` in the second annotator's separate copy. Open `http://127.0.0.1:8765`. The server stays local, uses Python's standard library and does not launch inference. Stop it with Ctrl-C. Read the guide before practice. Independent human ratings have not been created; the synthetic test fixtures are software tests only.

The final representative audit size awaits practice timing and coordinator decisions. The representative sampling command requires the complete final generation frame. The targeted queue additionally awaits actual canonical judge outputs. The scheduled heartbeat remains paused; this preparation does not restart it.
