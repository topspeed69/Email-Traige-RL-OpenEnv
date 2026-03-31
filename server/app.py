from fastapi import FastAPI, HTTPException
from .environment import EmailTriageEnv
from .models import Action, Observation, Reward
from .graders import grade_episode
from typing import Dict, Any

app = FastAPI()

# Global environment instance
env = EmailTriageEnv()

@app.post("/reset", response_model=Observation)
async def reset(task_id: str) -> Observation:
    """Reset environment with task"""
    try:
        observation = env.reset(task_id)
        return observation
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

@app.get("/state", response_model=Observation)
async def state() -> Observation:
    """Get current state without stepping"""
    return env.state()

@app.post("/step")
async def step(action: Action) -> Dict[str, Any]:
    """Execute step"""
    observation, reward, done, truncated, info = env.step(action)
    
    # If episode complete, calculate final score
    if done or truncated:
        final_score = grade_episode(env.current_task, env)
        info["final_score"] = final_score
    
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "truncated": truncated,
        "info": info
    }

@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok", "environment": "EmailTriageEnv"}

@app.get("/tasks")
async def list_tasks():
    """List available tasks"""
    return ["easy", "medium", "hard"]

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
