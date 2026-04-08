"""Quick test: WebSocket connection to running server."""
import asyncio
from openenv.core import EnvClient
from openenv.core.client_types import StepResult


class TestClient(EnvClient[dict, dict, dict]):
    def _step_payload(self, action):
        return action

    def _parse_result(self, payload):
        obs = payload.get("observation", {})
        reward_data = payload.get("reward", {})
        reward = (
            reward_data.get("total", 0.0)
            if isinstance(reward_data, dict)
            else float(reward_data or 0)
        )
        return StepResult(
            observation=obs,
            reward=reward,
            done=payload.get("done", False) or payload.get("truncated", False),
        )

    def _parse_state(self, payload):
        return payload


async def test():
    client = TestClient(base_url="http://localhost:8000")
    await client.connect()
    result = await client.reset(task_id="easy")
    inbox_count = len(result.observation.get("inbox", []))
    print(f"Reset OK: inbox={inbox_count}, done={result.done}")

    # Take one step
    action = {"action_type": "skip", "email_id": "none"}
    result = await client.step(action)
    print(f"Step OK: reward={result.reward}, done={result.done}")

    await client.close()
    print("WebSocket test PASSED")


if __name__ == "__main__":
    asyncio.run(test())
