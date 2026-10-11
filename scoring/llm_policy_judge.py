"""
scoring/llm_policy_judge.py

Run three independent first-pass judgments for each model output via
the vLLM OpenAI-compatible backend.  One retry per failed call.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional, Tuple

from models.vllm_client import ServerUnavailableError
from scoring.llm_policy_schema import validate_judge_output, JUDGE_RESPONSE_FORMAT
from scoring.llm_policy_prompts import build_judge_user_message

# Constrain field names, types, and labels; validation still catches truncation
# and any server-side failure to honor the requested schema.
_JUDGE_RESPONSE_FORMAT = JUDGE_RESPONSE_FORMAT

logger = logging.getLogger(__name__)

JUDGE_IDS = ("A", "B", "C")


def _row_hash(model_output: str) -> str:
    return hashlib.sha256(model_output.encode("utf-8")).hexdigest()


def _call_once(
    client,
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Single LLM call.  Returns (parsed_record, parse_error).
    parse_error is None on success.
    """
    try:
        raw, _ = client.generate(
            system_prompt=system_prompt,
            user_prompt=user_message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=_JUDGE_RESPONSE_FORMAT,
        )
    except ServerUnavailableError:
        raise
    except Exception as e:
        return None, f"LLM call error: {e}"

    record, err = validate_judge_output(raw)
    return record, err


def judge_row(
    *,
    row_index: int,
    job_id: str,
    model_output: str,
    prompts: Dict[str, str],  # {"A": ..., "B": ..., "C": ...}
    client,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 250,
) -> List[Dict]:
    """
    Run three independent first-pass judgments for *model_output*.

    Returns a list of vote dicts (one per judge variant).
    Each vote dict contains all schema fields plus bookkeeping columns:
      job_id, row_index, row_hash, judge_id, judge_prompt_variant,
      judge_model, raw_json, parse_error.
    """
    row_hash = _row_hash(model_output)
    user_message = build_judge_user_message(model_output)

    votes: List[Dict] = []
    for judge_id in JUDGE_IDS:
        system_prompt = prompts[judge_id]
        record, err = _call_once(
            client, system_prompt, user_message, model, temperature, max_tokens
        )

        if err and record is None:
            # One retry
            logger.warning("Judge %s row %d: first attempt failed (%s). Retrying.", judge_id, row_index, err)
            record, err = _call_once(
                client, system_prompt, user_message, model, temperature, max_tokens
            )

        vote: Dict = {
            "job_id": job_id,
            "row_index": row_index,
            "row_hash": row_hash,
            "judge_id": judge_id,
            "judge_prompt_variant": judge_id,
            "judge_model": model,
            "raw_json": "",
            "parse_error": err or "",
        }

        if record is not None:
            vote.update(record)
        else:
            # Fill sentinel values so the vote row is always structurally complete
            vote.update({
                "primary_label": "parse_error",
                "secondary_label": None,
                "confidence": 0.0,
                "contains_answer": False,
                "contains_refusal": False,
                "contains_clarifying_question": False,
                "contains_safe_redirect": False,
                "mentions_instruction_priority": False,
                "treats_external_text_as_data": False,
                "evidence": "",
                "reason": "",
            })

        votes.append(vote)

    return votes
