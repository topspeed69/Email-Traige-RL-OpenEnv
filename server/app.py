from fastapi import FastAPI, HTTPException
from .environment import EmailTriageEnv
from .models import Action, Observation, Reward
from .graders import grade_episode
from typing import Dict, Any

app = FastAPI()

# Global environment instance
env = EmailTriageEnv()

@app.post("/reset", response_model=Observation)
async def reset(task_id: str = "easy") -> Observation:
    """Reset environment with task"""
    try:
        observation = env.reset(task_id)
        return observation
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

@app.get("/state")
async def state() -> Dict[str, Any]:
    """Get current state without stepping"""
    return {
        "episode_id": env.current_task,
        "step_count": env.current_step
    }

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

from fastapi.responses import HTMLResponse

@app.get("/")
@app.get("/web")
async def root():
    """Beautiful landing page for Hugging Face Spaces"""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Email Triage RL Environment</title>
            <style>
                body { font-family: 'Inter', system-ui, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background: #0f172a; color: #f8fafc; margin: 0; }
                .container { text-align: center; max-width: 600px; padding: 2.5rem; background: #1e293b; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); }
                h1 { color: #38bdf8; margin-bottom: 1rem; font-size: 2rem; }
                p { line-height: 1.6; color: #cbd5e1; font-size: 1.1rem; margin-bottom: 2rem; }
                .code { background: #0f172a; padding: 1.25rem; border-radius: 10px; border: 1px solid #334155; text-align: left; font-family: monospace; font-size: 0.95rem; color: #10b981; }
                .success { display: inline-block; background: rgba(52, 211, 153, 0.1); color: #34d399; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem; font-weight: 600; margin-bottom: 1rem; }
            </style>
        </head>
        <body>
            <div class="container">
                <span class="success">● System Online</span>
                <h1>📧 Email Triage RL</h1>
                <p>Headless OpenEnv agent sandbox is successfully deployed and ready for API interaction! Run your agent against this cloud deployment locally.</p>
                <div class="code">
                    <span style="color: #64748b;"># Set cloud endpoint</span><br/>
                    export ENV_URL="https://speedbuoy69-email-triage-env.hf.space"<br/><br/>
                    <span style="color: #64748b;"># Run your agent evaluation</span><br/>
                    python inference.py
                </div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "environment": "EmailTriageEnv"}

@app.get("/tasks")
async def list_tasks():
    """List available tasks"""
    return ["easy", "medium", "hard"]

from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Fully compliant WebSocket endpoint for OpenEnv generic client."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            msg_data = data.get("data", {})
            
            if msg_type == "reset":
                task_id = msg_data.get("task_id", "easy")
                try:
                    obs = env.reset(task_id)
                except Exception as e:
                    await websocket.send_json({"type": "error", "data": {"message": str(e), "code": "RESET_FAILED"}})
                    continue
                    
                await websocket.send_json({
                    "data": {
                        "observation": obs.model_dump(),
                        "reward": {"total": 0.0},
                        "done": False,
                        "truncated": False,
                        "info": {}
                    }
                })
                
            elif msg_type == "step":
                try:
                    action = Action(**msg_data)
                    obs, reward, done, trunc, info = env.step(action)
                    if done or trunc:
                        info["final_score"] = grade_episode(env.current_task, env)
                        
                    await websocket.send_json({
                        "data": {
                            "observation": obs.model_dump(),
                            "reward": reward.model_dump(),
                            "done": done,
                            "truncated": trunc,
                            "info": info
                        }
                    })
                except Exception as e:
                    await websocket.send_json({"type": "error", "data": {"message": str(e), "code": "STEP_FAILED"}})
                    
            elif msg_type == "state":
                await websocket.send_json({
                    "data": {
                        "episode_id": env.current_task,
                        "step_count": env.current_step
                    }
                })
                
            elif msg_type == "close":
                break
    except WebSocketDisconnect:
        pass


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()
