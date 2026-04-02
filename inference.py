import os
import json
import re
import requests
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Environment variables (required)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

ENV_URL = "http://localhost:8000"  # FastAPI default port or change to match your deployment

# Check if we should use local LLM API for test
if not API_KEY and not API_BASE_URL:
    client = None
else:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an email triage AI. You make ONE atomic action per step.

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

def _decide_next_action_heuristic(obs):
    """Deterministic fallback: process emails through the multi-step pipeline."""
    
    # Priority 1: Advance in-progress emails to completion
    for ep in obs.get("in_progress", []):
        eid = ep["id"]
        body = ep.get("body") or ""
        subject = ep.get("subject") or ""
        thread_ctx = ep.get("thread_context") or ""
        text = (subject + " " + body + " " + str(thread_ctx)).lower()
        
        if not ep.get("priority_set"):
            # Needs priority
            priority = "high" if "asap" in text or "urgent" in text else "medium"
            return {"action_type": "set_priority", "email_id": eid, "priority": priority}
            
        # Has category + priority, needs terminal action
        cat = ep.get("category_set", "")
        if cat in ("spam", "general_info"):
            return {"action_type": "archive", "email_id": eid}
        elif cat == "urgent_escalation":
            return {"action_type": "escalate", "email_id": eid}
        else:
            # Route based on category
            team_map = {
                "billing_issue": "finance",
                "technical_support": "engineering",
                "meeting_request": "support",
                "sales_inquiry": "sales",
                "internal": "support",
            }
            team = team_map.get(cat, "support")
            return {"action_type": "route", "email_id": eid, "team": team}
    
    # Priority 2: Start classifying new inbox emails
    inbox = obs.get("inbox", [])
    if inbox:
        e = inbox[0]
        eid = e["id"]
        
        # If threaded and thread NOT yet read, read it first
        if e.get("thread_id") and not e.get("thread_read"):
            return {"action_type": "read_thread", "email_id": eid}
            
        body = e.get("body") or ""
        subject = e.get("subject") or ""
        thread_ctx = e.get("thread_context") or ""
        text = (subject + " " + body + " " + str(thread_ctx)).lower()
        
        # Simple heuristics for category
        if "invoice" in text or "billing" in text or "charge" in text or "refund" in text or "payment" in text:
            category = "billing_issue"
        elif "asap" in text or "urgent" in text:
            category = "urgent_escalation"
        elif "crash" in text or "error" in text or "failing" in text or "bug" in text:
            category = "technical_support"
        elif "meeting" in text or "sync" in text or "call" in text:
            category = "meeting_request"
        elif "quote" in text or "pricing" in text or "sales" in text or "enterprise" in text:
            category = "sales_inquiry"
        else:
            category = "general_info"
            
        return {"action_type": "classify", "email_id": eid, "category": category}
    
    return {"action_type": "skip", "email_id": "none"}


def run_task(task_id: str, verbose: bool = False) -> float:
    """Run inference on one task"""
    print(f"\n{'='*50}")
    print(f"Running task: {task_id}")
    print(f"{'='*50}\n")
    
    # Reset
    try:
        resp = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
        resp.raise_for_status()
        obs = resp.json()
    except Exception as e:
        print(f"Failed to connect to environment server: {e}")
        return 0.0
    
    done = False
    truncated = False
    step_count = 0
    final_score = 0.0
    action_history = []
    
    while not (done or truncated):
        step_count += 1
        
        # Build prompt - SAFE SLICING WITH (val or '')[:n] PATTERN
        inbox_text = "\n\n".join([
            f"Email [ID: {e['id']}]\n"
            f"From: {e['sender']}\n"
            f"Subject: {(e.get('subject') or '')[:100]}\n"
            f"Body: {(e.get('body') or '')[:150]}...\n"
            + (f"Thread Context: {(e.get('thread_context') or '')[:200]}...\n" if e.get('thread_context') else "")
            + f"Thread: {e.get('thread_id', 'none')}"
            + (f" (thread {'READ' if e.get('thread_read') else 'UNREAD — must read_thread first'})" if e.get('thread_id') else "")
            for e in obs['inbox'][:5]
        ])
        
        in_progress_text = "\n\n".join([
            f"In-Progress [ID: {e['id']}]\n"
            f"Subject: {(e.get('subject') or '')[:100]}\n"
            f"Body/Thread Context: {(e.get('body') or '')[:50]}... {(e.get('thread_context') or '')[:50]}\n"
            f"Category: {e.get('category_set', 'NOT SET')}\n"
            f"Priority: {'SET' if e.get('priority_set') else 'NOT SET'}\n"
            f"Disposition: {e.get('disposition', 'PENDING')}\n"
            f"NEXT ACTION NEEDED: {'set_priority' if not e.get('priority_set') else ('route/archive/escalate' if e.get('category_set') else 'classify')}"
            for e in obs.get('in_progress', [])[:5]
        ])
        
        user_prompt = f"""Inbox ({len(obs['inbox'])} unread):

{inbox_text}

In-Progress ({len(obs.get('in_progress', []))} emails):

{in_progress_text}

Recent Actions:
{', '.join(action_history[-3:]) if action_history else 'None'}

Step {obs['current_step']}/{obs['max_steps']}
Fully processed: {obs['processed_count']}
SLA violations: {obs['sla_violations']}

Choose ONE action. Respond with VALID JSON ONLY - no explanation, no markdown, just the JSON object."""
        
        # Call LLM
        used_method = "Unknown"
        try:
            if client is not None:
                used_method = "LLM"
                # NVIDIA API doesn't support system role, so include system prompt in user message
                full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=256,
                )
                
                response_text = completion.choices[0].message.content.strip()
                
                # More robust JSON extraction
                action = None
                
                # Try 1: Direct JSON parse
                try:
                    action = json.loads(response_text)
                except json.JSONDecodeError:
                    pass
                
                # Try 2: If response looks like JSON content without braces, wrap it
                if action is None and '"action_type"' in response_text:
                    try:
                        wrapped = "{" + response_text + "}"
                        action = json.loads(wrapped)
                    except json.JSONDecodeError:
                        pass
                
                # Try 3: Extract JSON object with proper bracket matching
                if action is None:
                    start_idx = response_text.find('{')
                    if start_idx != -1:
                        # Find matching closing brace
                        brace_count = 0
                        for i in range(start_idx, len(response_text)):
                            if response_text[i] == '{':
                                brace_count += 1
                            elif response_text[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = response_text[start_idx:i+1]
                                    try:
                                        action = json.loads(json_str)
                                    except json.JSONDecodeError:
                                        pass
                                    break
                
                # Try 4: Use regex as last resort
                if action is None:
                    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
                    if match:
                        try:
                            action = json.loads(match.group(0))
                        except json.JSONDecodeError:
                            pass
                
                if action is None:
                    raise ValueError(f"Could not extract valid JSON from response: {response_text[:100]}")
                    
            else:
                used_method = "Heuristic (no client)"
                # Deterministic heuristic fallback
                action = _decide_next_action_heuristic(obs)
            
        except Exception as e:
            if verbose:
                print(f"  [JSON parse error: {e}]")
            used_method = "Heuristic (fallback due to LLM error)"
            action = _decide_next_action_heuristic(obs)
        
        # Execute step
        resp = requests.post(f"{ENV_URL}/step", json=action)
        
        if resp.status_code != 200:
            print(f"Action rejected (Status {resp.status_code}): {resp.text}")
            used_method = "Heuristic (fallback due to rejection)"
            action = _decide_next_action_heuristic(obs)
            resp = requests.post(f"{ENV_URL}/step", json=action)

        result = resp.json()
        
        obs = result['observation']
        reward = result['reward']['total']
        done = result['done']
        truncated = result['truncated']
        
        action_history.append(f"{action.get('action_type', 'unknown')} on {action.get('email_id', 'N/A')}")
        if len(action_history) > 3:
            action_history.pop(0)
        
        log_msg = f"Step {step_count}: {action['action_type']:15s} on {action.get('email_id', 'N/A'):12s} -> reward: {reward:+.2f}"
        if verbose:
            log_msg += f" [{used_method}]"
        print(log_msg)
        
        if step_count > 500:  # Safety break (higher due to multi-step)
            break
            
        final_score = result['info'].get('final_score', 0.0)
    
    emails_done = result['info'].get('emails_done', 0)
    print(f"\nTask complete! Emails fully done: {emails_done} | Final score: {final_score:.3f}\n")
    
    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run email triage inference")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs, including whether LLM or Heuristic is used")
    args = parser.parse_args()

    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = run_task(task, verbose=args.verbose)
    
    print("\n" + "="*50)
    print("BASELINE RESULTS")
    print("="*50)
    for task, score in scores.items():
        print(f"{task:10s}: {score:.3f}")