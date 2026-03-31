from fastapi.testclient import TestClient
from server.app import app
from server.models import Action

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_list_tasks():
    response = client.get("/tasks")
    assert response.status_code == 200
    assert "easy" in response.json()

def test_reset_and_multistep():
    # Test reset
    response = client.post("/reset", params={"task_id": "easy"})
    assert response.status_code == 200
    obs = response.json()
    assert "inbox" in obs
    assert "in_progress" in obs
    assert len(obs["inbox"]) > 0
    
    # Step 1: classify
    email_id = obs["inbox"][0]["id"]
    response = client.post("/step", json={
        "action_type": "classify",
        "email_id": email_id,
        "category": "general_info",
    })
    assert response.status_code == 200
    res = response.json()
    assert "reward" in res
    assert "observation" in res
    assert len(res["observation"]["in_progress"]) == 1

    # Step 2: set_priority
    response = client.post("/step", json={
        "action_type": "set_priority",
        "email_id": email_id,
        "priority": "low",
    })
    assert response.status_code == 200

    # Step 3: archive
    response = client.post("/step", json={
        "action_type": "archive",
        "email_id": email_id,
    })
    assert response.status_code == 200
    res = response.json()
    assert res["observation"]["processed_count"] >= 1
