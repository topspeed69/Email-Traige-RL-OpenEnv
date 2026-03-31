import os
import json
import requests
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
    pass
else:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an email triage AI. You make ONE atomic action per step.

Available action types:
  - read_thread   : Read thread context (required before classifying threaded emails)
  - classify      : Set category ONLY.  Categories: spam, billing_issue, technical_support, meeting_request, sales_inquiry, urgent_escalation, general_info, internal
  - set_priority  : Set priority ONLY.   Priorities: high, medium, low
  - route         : Route to a team.     Teams: engineering, finance, sales, support
  - archive       : Archive the email.   Best for spam/general_info
  - escalate      : Escalate the email.  Best for urgent_escalation
  - skip          : Do nothing this step

Workflow per email:
  1. (If threaded) read_thread first
  2. classify with a category
  3. set_priority
  4. route / archive / escalate (terminal)

Respond ONLY with valid JSON for ONE action. Examples:

{"action_type": "classify", "email_id": "email_001", "category": "billing_issue"}
{"action_type": "set_priority", "email_id": "email_001", "priority": "high"}
{"action_type": "route", "email_id": "email_001", "team": "finance"}
{"action_type": "archive", "email_id": "email_003"}
{"action_type": "escalate", "email_id": "email_005"}
"""

def _decide_next_action_heuristic(obs):
    """Deterministic fallback: process emails through the multi-step pipeline."""
    
    # Priority 1: Advance in-progress emails to completion
    for ep in obs.get("in_progress", []):
        eid = ep["id"]
        text = (ep.get("subject", "") + " " + ep.get("body", "") + " " + str(ep.get("thread_context", ""))).lower()
        
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
            
        text = (e.get("subject", "") + " " + e.get("body", "") + " " + str(e.get("thread_context", ""))).lower()
        
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


def run_task(task_id: str) -> float:
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
    
    while not (done or truncated):
        step_count += 1
        
        # Build prompt
        inbox_text = "\n\n".join([
            f"Email [ID: {e['id']}]\n"
            f"From: {e['sender']}\n"
            f"Subject: {e['subject']}\n"
            f"Body: {e['body'][:150]}...\n"
            f"Thread: {e.get('thread_id', 'none')}"
            + (f" (thread {'READ' if e.get('thread_read') else 'UNREAD — must read_thread first'})" if e.get('thread_id') else "")
            for e in obs['inbox'][:5]
        ])
        
        in_progress_text = "\n\n".join([
            f"In-Progress [ID: {e['id']}]\n"
            f"Category: {e.get('category_set', 'NOT SET')}\n"
            f"Priority: {'SET' if e.get('priority_set') else 'NOT SET'}\n"
            f"Disposition: {e.get('disposition', 'PENDING')}"
            for e in obs.get('in_progress', [])[:5]
        ])
        
        user_prompt = f"""Inbox ({len(obs['inbox'])} unread):

{inbox_text}

In-Progress ({len(obs.get('in_progress', []))} emails):

{in_progress_text}

Step {obs['current_step']}/{obs['max_steps']}
Fully processed: {obs['processed_count']}
SLA violations: {obs['sla_violations']}

Choose ONE action (JSON only):"""
        
        # Call LLM
        try:
            if 'client' in globals():
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                
                response_text = completion.choices[0].message.content
                response_text = response_text.replace("```json", "").replace("```", "").strip()
                action = json.loads(response_text)
            else:
                # Deterministic heuristic fallback
                action = _decide_next_action_heuristic(obs)
            
        except Exception as e:
            print(f"LLM error: {e}")
            action = _decide_next_action_heuristic(obs)
        
        # Execute step
        resp = requests.post(f"{ENV_URL}/step", json=action)
        
        if resp.status_code != 200:
            print(f"Action rejected (Status {resp.status_code}): {resp.text}")
            action = _decide_next_action_heuristic(obs)
            resp = requests.post(f"{ENV_URL}/step", json=action)

        result = resp.json()
        
        obs = result['observation']
        reward = result['reward']['total']
        done = result['done']
        truncated = result['truncated']
        
        print(f"Step {step_count}: {action['action_type']:15s} on {action.get('email_id', 'N/A'):12s} -> reward: {reward:+.2f}")
        
        if step_count > 500:  # Safety break (higher due to multi-step)
            break
            
        final_score = result['info'].get('final_score', 0.0)
    
    emails_done = result['info'].get('emails_done', 0)
    print(f"\nTask complete! Emails fully done: {emails_done} | Final score: {final_score:.3f}\n")
    
    return final_score

if __name__ == "__main__":
    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = run_task(task)
    
    print("\n" + "="*50)
    print("BASELINE RESULTS")
    print("="*50)
    for task, score in scores.items():
        print(f"{task:10s}: {score:.3f}")
