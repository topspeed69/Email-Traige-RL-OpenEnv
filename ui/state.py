import sys
import os
import streamlit as st

# Path setup so we can import 'server'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from server.environment import EmailTriageEnv
from server.models import Action
from server.graders import grade_episode
from server.tasks import TASK_CONFIGS

import requests
from server.models import Action, Observation, Reward

SERVER_URL = "http://localhost:7860"

def init_state():
    """Initialize all session state variables."""
    defaults = {
        "env": None,  # No longer holds the class, just the debug dict
        "obs": None,
        "task_id": "easy",
        "current_run_task": "easy", # To track internal level when "all" is selected
        "step_history": [],
        "reward_history": [],
        "cumulative_rewards": [],
        "action_log": [],
        "category_accuracy": {"correct": 0, "incorrect": 0},
        "priority_accuracy": {"correct": 0, "incorrect": 0},
        "routing_accuracy": {"correct": 0, "incorrect": 0},
        "is_running": False,
        "episode_done": False,
        "final_score": None,
        "auto_speed": 0.3,
        "mode": "heuristic",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _fetch_debug_state():
    try:
        r = requests.get(f"{SERVER_URL}/debug_state")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching debug state: {e}")
        return None


def reset_env(target_task=None):
    """Reset the environment over HTTP."""
    if st.session_state.task_id == "all":
        if target_task:
            st.session_state.current_run_task = target_task
        else:
            st.session_state.current_run_task = "easy"
    else:
        st.session_state.current_run_task = st.session_state.task_id

    try:
        r = requests.post(f"{SERVER_URL}/reset?task_id={st.session_state.current_run_task}")
        r.raise_for_status()
        obs_dict = r.json()
        st.session_state.obs = Observation(**obs_dict)
    except Exception as e:
        st.error(f"Failed to connect to backend {SERVER_URL}. Is server/app.py running? Error: {e}")
        return

    st.session_state.env = _fetch_debug_state()
    st.session_state.step_history = []
    st.session_state.reward_history = []
    st.session_state.cumulative_rewards = [0.0]
    st.session_state.action_log = []
    st.session_state.category_accuracy = {"correct": 0, "incorrect": 0}
    st.session_state.priority_accuracy = {"correct": 0, "incorrect": 0}
    st.session_state.routing_accuracy = {"correct": 0, "incorrect": 0}
    st.session_state.is_running = False
    st.session_state.episode_done = False
    st.session_state.final_score = None


def execute_step(action_dict: dict):
    """Execute step via HTTP backend."""
    if st.session_state.episode_done:
        return

    try:
        r = requests.post(f"{SERVER_URL}/step", json=action_dict)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        st.error(f"Action failed: {e}")
        return

    obs = Observation(**resp["observation"])
    reward = Reward(**resp["reward"])
    done = resp["done"]
    truncated = resp["truncated"]
    info = resp["info"]

    st.session_state.obs = obs
    st.session_state.env = _fetch_debug_state()

    st.session_state.reward_history.append(reward.total)
    prev_cum = st.session_state.cumulative_rewards[-1] if st.session_state.cumulative_rewards else 0.0
    st.session_state.cumulative_rewards.append(prev_cum + reward.total)

    if reward.category_correct > 0:
        st.session_state.category_accuracy["correct"] += 1
    elif reward.category_correct < 0:
        st.session_state.category_accuracy["incorrect"] += 1

    if reward.priority_correct > 0:
        st.session_state.priority_accuracy["correct"] += 1
    elif reward.priority_correct < 0:
        st.session_state.priority_accuracy["incorrect"] += 1

    if reward.routing_correct > 0:
        st.session_state.routing_accuracy["correct"] += 1
    elif reward.routing_correct < 0:
        st.session_state.routing_accuracy["incorrect"] += 1

    reward_color = "positive" if reward.total >= 0 else "negative"
    st.session_state.action_log.append({
        "step": st.session_state.env["current_step"],
        "action": action_dict,
        "reward": resp["reward"],
        "reward_color": reward_color,
        "error": obs.last_action_error,
        "done": done or truncated,
    })

    st.session_state.step_history.append({
        "step": st.session_state.env["current_step"],
        "inbox_size": len(obs.inbox),
        "in_progress": len(obs.in_progress),
        "processed": obs.processed_count,
        "sla_violations": obs.sla_violations,
    })

    if done or truncated:
        st.session_state.episode_done = True
        st.session_state.final_score = info.get("final_score", 0.0)
        st.session_state.is_running = False
