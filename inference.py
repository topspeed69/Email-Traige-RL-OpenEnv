"""
Inference Script — Email Triage RL Environment
===================================
MANDATORY
- API_BASE_URL, MODEL_NAME, HF_TOKEN environment variables
- Optional: LOCAL_IMAGE_NAME when using from_docker_image()
- OpenAI client for all LLM calls
- Stdout follows required [START]/[STEP]/[END] format
"""

import asyncio
import os
import json
import re
from typing import List, Optional, Dict, Any

from openai import OpenAI
from openenv.core import EnvClient
from openenv.core.client_types import StepResult

# ── Environment variables (matching sample inference.py exactly) ──────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://integrate.api.nvidia.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "mistralai/mistral-7b-instruct-v0.2"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "email-triage-env"
MAX_STEPS = 500


# ── Minimal OpenEnv client (avoids relative-import issues) ────────────────
class EmailEnvClient(EnvClient[dict, dict, dict]):
    """Thin OpenEnv client that keeps observations as plain dicts."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.last_info: Dict[str, Any] = {}

    def _step_payload(self, action: dict) -> dict:
        return action

    def _parse_result(self, payload: dict) -> StepResult[dict]:
        obs = payload.get("observation", {})
        reward_data = payload.get("reward", {})
        reward = (
            reward_data.get("total", 0.0)
            if isinstance(reward_data, dict)
            else float(reward_data or 0)
        )
        self.last_info = payload.get("info", {})
        return StepResult(
            observation=obs,
            reward=reward,
            done=payload.get("done", False) or payload.get("truncated", False),
        )

    def _parse_state(self, payload: dict) -> dict:
        return payload


# ── Required stdout helpers ───────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an email triage AI. You make ONE atomic action per step.

Available action types:
  - read_thread   : Read thread context (required before classifying threaded emails)
  - classify      : Set category ONLY. Categories: spam, billing_issue, technical_support, meeting_request, sales_inquiry, urgent_escalation, general_info, internal
  - set_priority  : Set priority ONLY. Priorities: high, medium, low
  - route         : Route to a team. Teams EXACTLY ONE OF: engineering, finance, sales, support (DO NOT use "general")
  - archive       : Archive the email. Best for spam/general_info
  - escalate      : Escalate the email. Best for urgent_escalation
  - skip          : Do nothing this step

WORKFLOW PER EMAIL (follow this sequence):
  1. (If threaded) read_thread first
  2. classify with a category
  3. set_priority
  4. route / archive / escalate (terminal action - email is done after this)

CRITICAL RULES:
- If a thread is marked READ, DO NOT read_thread again. Move on to classify.
- In-Progress emails need missing fields filled (Priority, Category) and then a terminal action.
- Do NOT repeat the exact same action if it was just done.
- NEVER classify an email that already has a category set.
- NEVER set_priority on an email that already has priority set.
- Once an email has category + priority, it needs a terminal action (route/archive/escalate).

Examples of complete sequences:
- New email: classify -> set_priority -> route
- Threaded email: read_thread -> classify -> set_priority -> route
- Spam: classify -> set_priority -> archive
- Urgent: classify -> set_priority -> escalate

IMPORTANT: Your response MUST be valid JSON. Do not include any text before or after the JSON.

JSON Format Examples:
{"action_type": "classify", "email_id": "email_001", "category": "billing_issue"}
{"action_type": "set_priority", "email_id": "email_001", "priority": "high"}
{"action_type": "route", "email_id": "email_001", "team": "finance"}
{"action_type": "archive", "email_id": "email_003"}
{"action_type": "escalate", "email_id": "email_005"}
{"action_type": "read_thread", "email_id": "email_002"}"""


# ── Heuristic fallback ───────────────────────────────────────────────────
def _decide_next_action_heuristic(obs: dict) -> dict:
    """Deterministic fallback: process emails through the multi-step pipeline."""
    for ep in obs.get("in_progress", []):
        eid = ep["id"]
        text = (
            (ep.get("subject") or "")
            + " " + (ep.get("body") or "")
            + " " + str(ep.get("thread_context") or "")
        ).lower()

        if not ep.get("priority_set"):
            priority = "high" if ("asap" in text or "urgent" in text) else "medium"
            return {"action_type": "set_priority", "email_id": eid, "priority": priority}

        cat = ep.get("category_set", "")
        if cat in ("spam", "general_info"):
            return {"action_type": "archive", "email_id": eid}
        if cat == "urgent_escalation":
            return {"action_type": "escalate", "email_id": eid}
        team_map = {
            "billing_issue": "finance",
            "technical_support": "engineering",
            "meeting_request": "support",
            "sales_inquiry": "sales",
            "internal": "support",
        }
        return {"action_type": "route", "email_id": eid, "team": team_map.get(cat, "support")}

    inbox = obs.get("inbox", [])
    if inbox:
        e = inbox[0]
        eid = e["id"]
        if e.get("thread_id") and not e.get("thread_read"):
            return {"action_type": "read_thread", "email_id": eid}
        text = (
            (e.get("subject") or "")
            + " " + (e.get("body") or "")
            + " " + str(e.get("thread_context") or "")
        ).lower()
        if any(w in text for w in ("invoice", "billing", "charge", "refund", "payment")):
            category = "billing_issue"
        elif any(w in text for w in ("asap", "urgent")):
            category = "urgent_escalation"
        elif any(w in text for w in ("crash", "error", "failing", "bug")):
            category = "technical_support"
        elif any(w in text for w in ("meeting", "sync", "call")):
            category = "meeting_request"
        elif any(w in text for w in ("quote", "pricing", "sales", "enterprise")):
            category = "sales_inquiry"
        else:
            category = "general_info"
        return {"action_type": "classify", "email_id": eid, "category": category}

    return {"action_type": "skip", "email_id": "none"}


# ── JSON extraction helpers ──────────────────────────────────────────────
def _extract_json(text: str) -> Optional[dict]:
    """Try multiple strategies to extract JSON from LLM response."""
    # Try 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try 2: Find first {...} block
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    # Try 3: Regex
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── LLM action selection ─────────────────────────────────────────────────
def get_llm_action(
    llm_client: OpenAI, obs: dict, action_history: List[str], verbose: bool = False
) -> tuple[dict, str]:
    """Call the LLM for the next action. Returns (action_dict, method_label)."""

    inbox_text = "\n\n".join(
        [
            f"Email [ID: {e['id']}]\n"
            f"From: {e['sender']}\n"
            f"Subject: {(e.get('subject') or '')[:100]}\n"
            f"Body: {(e.get('body') or '')[:150]}...\n"
            + (
                f"Thread Context: {(e.get('thread_context') or '')[:200]}...\n"
                if e.get("thread_context")
                else ""
            )
            + f"Thread: {e.get('thread_id', 'none')}"
            + (
                f" (thread {'READ' if e.get('thread_read') else 'UNREAD — must read_thread first'})"
                if e.get("thread_id")
                else ""
            )
            for e in obs.get("inbox", [])[:5]
        ]
    )

    in_progress_text = "\n\n".join(
        [
            f"In-Progress [ID: {e['id']}]\n"
            f"Subject: {(e.get('subject') or '')[:100]}\n"
            f"Category: {e.get('category_set', 'NOT SET')}\n"
            f"Priority: {'SET' if e.get('priority_set') else 'NOT SET'}\n"
            f"Disposition: {e.get('disposition', 'PENDING')}\n"
            f"NEXT: {'set_priority' if not e.get('priority_set') else 'route/archive/escalate'}"
            for e in obs.get("in_progress", [])[:5]
        ]
    )

    user_prompt = (
        f"Inbox ({len(obs.get('inbox', []))} unread):\n\n{inbox_text}\n\n"
        f"In-Progress ({len(obs.get('in_progress', []))} emails):\n\n{in_progress_text}\n\n"
        f"Recent Actions: {', '.join(action_history[-3:]) if action_history else 'None'}\n"
        f"Step {obs['current_step']}/{obs['max_steps']}\n"
        f"Fully processed: {obs['processed_count']}\n"
        f"SLA violations: {obs['sla_violations']}\n\n"
        f"Choose ONE action. Respond with VALID JSON ONLY."
    )

    try:
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        completion = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.2,
            max_tokens=256,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        action = _extract_json(response_text)
        if action is None:
            raise ValueError(f"No valid JSON: {response_text[:100]}")
        return action, "LLM"
    except Exception as e:
        if verbose:
            print(f"  [DEBUG] LLM error: {e}", flush=True)
        return _decide_next_action_heuristic(obs), "Heuristic(fallback)"


# ── Single task runner ────────────────────────────────────────────────────
async def run_task(
    env: EmailEnvClient, llm_client: OpenAI, task_id: str, verbose: bool = False
) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        action_history: List[str] = []

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action, method = get_llm_action(llm_client, obs, action_history, verbose)

            try:
                result = await env.step(action)
                error_from_obs = result.observation.get("last_action_error")
                if error_from_obs and method == "LLM":
                    raise ValueError(f"Action rejected by env: {error_from_obs}")
            except Exception as e:
                # Pydantic validation errors or env rejections trigger heuristic fallback
                if method == "LLM":
                    if verbose:
                        print(f"  [DEBUG] LLM action invalid, falling back. Error: {e}", flush=True)
                    action = _decide_next_action_heuristic(obs)
                    result = await env.step(action)
                else:
                    raise e
            
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = obs.get("last_action_error")
            
            rewards.append(reward)
            steps_taken = step

            action_str = f"{action.get('action_type', 'unknown')}({action.get('email_id', 'N/A')})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            action_history.append(action_str)
            if len(action_history) > 3:
                action_history.pop(0)

            if done:
                break

        # Prefer server-computed final_score; fall back to reward-based
        final_score = env.last_info.get("final_score", None)
        if final_score is not None:
            score = float(final_score)
        elif rewards:
            score = max(sum(rewards) / len(rewards), 0.0)
        score = min(max(score, 0.0), 1.0)
        success = score > 0.1

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ──────────────────────────────────────────────────────────────────
async def main() -> None:
    # Create OpenAI LLM client using injected proxy vars
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Create environment client
    if LOCAL_IMAGE_NAME:
        env = await EmailEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env_url = os.getenv("ENV_URL", "http://localhost:8000")
        env = EmailEnvClient(base_url=env_url)
        await env.connect()

    try:
        scores: dict[str, float] = {}
        for task in ["easy", "medium", "hard"]:
            scores[task] = await run_task(env, llm_client, task)

        print("\n" + "=" * 50)
        print("BASELINE RESULTS")
        print("=" * 50)
        for task, s in scores.items():
            print(f"{task:10s}: {s:.3f}")
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())