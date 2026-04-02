from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import Action, Observation

class EmailTriageEnv(EnvClient[Action, Observation, State]):
    """
    Client for the Email Triage Environment.
    """

    def _step_payload(self, action: Action) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[Observation]:
        obs_data = payload.get("observation", {})
        observation = Observation(**obs_data)
        
        reward_data = payload.get("reward", {})
        if isinstance(reward_data, dict):
            reward = reward_data.get("total", 0.0)
        else:
            reward = float(reward_data)
            
        return StepResult(
            observation=observation,
            reward=reward,
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
