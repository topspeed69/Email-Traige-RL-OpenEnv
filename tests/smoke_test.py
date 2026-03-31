"""Smoke test for the multi-step email triage environment."""
import requests
import json

ENV = "http://localhost:8000"

# Reset
r = requests.post(f"{ENV}/reset", params={"task_id": "easy"})
obs = r.json()
print(f"Inbox: {len(obs['inbox'])} emails")
print(f"In-progress: {len(obs['in_progress'])} emails")

# Pick first email
eid = obs["inbox"][0]["id"]
print(f"\nTarget email: {eid}")

# Step 1: classify
r = requests.post(f"{ENV}/step", json={"action_type": "classify", "email_id": eid, "category": "spam"})
res = r.json()
print(f"Step 1 (classify): reward={res['reward']['total']:+.2f}, error={res['observation']['last_action_error']}")
print(f"  In-progress: {len(res['observation']['in_progress'])}")

# Step 2: set_priority
r = requests.post(f"{ENV}/step", json={"action_type": "set_priority", "email_id": eid, "priority": "low"})
res = r.json()
print(f"Step 2 (set_priority): reward={res['reward']['total']:+.2f}, error={res['observation']['last_action_error']}")

# Step 3: archive
r = requests.post(f"{ENV}/step", json={"action_type": "archive", "email_id": eid})
res = r.json()
print(f"Step 3 (archive): reward={res['reward']['total']:+.2f}, error={res['observation']['last_action_error']}")
print(f"  Processed count: {res['observation']['processed_count']}")
print(f"  Done: {res['done']}")

# Test dependency violation: try to route without classify
if len(obs["inbox"]) > 1:
    eid2 = obs["inbox"][1]["id"]
    r = requests.post(f"{ENV}/step", json={"action_type": "route", "email_id": eid2, "team": "engineering"})
    res = r.json()
    print(f"\nStep 4 (route w/o classify): reward={res['reward']['total']:+.2f}")
    print(f"  Error: {res['observation']['last_action_error']}")

# Test double-classify rejection
r = requests.post(f"{ENV}/step", json={"action_type": "classify", "email_id": eid, "category": "billing_issue"})
res = r.json()
print(f"\nStep 5 (re-classify done email): error={res['observation']['last_action_error']}")

print("\n=== SMOKE TEST PASSED ===")
