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

def test_reset_and_step():
    # Test reset
    response = client.post("/reset", params={"task_id": "easy"})
    assert response.status_code == 200
    obs = response.json()
    assert "inbox" in obs
    assert len(obs["inbox"]) > 0
    
    # Test step
    email_id = obs["inbox"][0]["id"]
    action = {
        "action_type": "classify",
        "email_id": email_id,
        "category": "general_info",
        "priority": "low",
        "team": "support"
    }
    
    response = client.post("/step", json=action)
    assert response.status_code == 200
    res = response.json()
    
    assert "reward" in res
    assert "observation" in res
    assert "done" in res
