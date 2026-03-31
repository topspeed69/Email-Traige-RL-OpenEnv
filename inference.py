import os
import json
import requests
from openai import OpenAI

# Environment variables (required)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

ENV_URL = "http://localhost:8000"  # FastAPI default port or change to match your deployment

# Check if we should use local LLM API for test
if not API_KEY and not API_BASE_URL:
    # Use dummy responses for the sake of the script working without config
    pass
else:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an email triage AI. Classify emails into categories, assign priorities, and route to teams.

Categories: spam, billing_issue, technical_support, meeting_request, sales_inquiry, urgent_escalation, general_info, internal
Priorities: high, medium, low
Teams: engineering, finance, sales, support

Respond ONLY with valid JSON:
{
  "action_type": "classify",
  "email_id": "email_001",
  "category": "billing_issue",
  "priority": "high",
  "team": "finance"
}"""

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
            f"Email {i+1} [ID: {e['id']}]\n"
            f"From: {e['sender']}\n"
            f"Subject: {e['subject']}\n"
            f"Body: {e['body'][:150]}..."
            for i, e in enumerate(obs['inbox'][:5])  # Show max 5
        ])
        
        user_prompt = f"""Inbox ({len(obs['inbox'])} unprocessed):

{inbox_text}

Step {obs['current_step']}/{obs['max_steps']}
Processed: {obs['processed_count']}
SLA violations: {obs['sla_violations']}

Choose next action (JSON only):"""
        
        # Call LLM
        try:
            if 'client' in globals():
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                
                response_text = completion.choices[0].message.content
                action = json.loads(response_text)
            else:
                # Dummy block if no OpenAI config for testing compilation
                if obs['inbox']:
                    action = {
                        "action_type": "classify",
                        "email_id": obs['inbox'][0]['id'],
                        "category": "general_info",
                        "priority": "medium",
                        "team": "support"
                    }
                else:
                    action = {"action_type": "skip", "email_id": "none"}
            
        except Exception as e:
            print(f"LLM error: {e}")
            # Fallback: classify first email as general_info
            if obs['inbox']:
                action = {
                    "action_type": "classify",
                    "email_id": obs['inbox'][0]['id'],
                    "category": "general_info",
                    "priority": "medium",
                    "team": "support"
                }
            else:
                action = {"action_type": "skip", "email_id": "none"}
        
        # Execute step
        resp = requests.post(f"{ENV_URL}/step", json=action)
        result = resp.json()
        
        obs = result['observation']
        reward = result['reward']['total']
        done = result['done']
        truncated = result['truncated']
        
        print(f"Step {step_count}: {action['action_type']} on {action.get('email_id', 'N/A')} -> reward: {reward:+.2f}")
        
        if step_count > 100:  # Safety break
            break
            
        final_score = result['info'].get('final_score', 0.0)
    
    print(f"\nTask complete! Final score: {final_score:.3f}\n")
    
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
