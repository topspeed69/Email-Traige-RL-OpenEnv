"""
Email Triage RL Environment — Interactive Streamlit Dashboard
=============================================================
A premium, interactive visualization of the EmailTriageEnv RL environment.
Demonstrates environment mechanics, agent decision-making, and reward signals.
"""

import sys
import os
import streamlit as st

# Path setup is duplicated here just in case, but ui.state also sets it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.state import init_state
from ui.utils import load_css
from ui.components import (
    render_sidebar,
    render_landing_page,
    render_top_metrics_bar,
    render_inbox_column,
    render_charts_and_accuracy,
    render_action_log,
    render_episode_completion,
    render_manual_controls,
    run_agent_loop,
)

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Triage RL Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS styles
load_css()
# Initialize session state
init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────
render_sidebar()

# ── Main Content ──────────────────────────────────────────────────────────
if st.session_state.env is None:
    render_landing_page()
    st.stop()

@st.fragment
def dashboard_fragment():
    # Active episode variables
    env = st.session_state.env
    obs = st.session_state.obs
    obs_dict = obs.model_dump()

    # Top Navigation
    render_top_metrics_bar(env, obs)

    # 3-column layout definition
    left_col, mid_col, right_col = st.columns([1.1, 1.4, 1.0])

    render_inbox_column(left_col, obs)
    render_charts_and_accuracy(mid_col, env)
    render_action_log(right_col)

    # Render episode completion summary banner
    if st.session_state.episode_done:
        render_episode_completion(env, obs)

    # Step execution controls (bottom)
    if not st.session_state.episode_done:
        st.markdown("---")
        
        if st.session_state.mode in ["heuristic", "llm_agent"]:
            run_agent_loop(env, obs_dict)
        elif st.session_state.mode == "manual":
            with right_col:
                render_manual_controls(obs)

dashboard_fragment()
