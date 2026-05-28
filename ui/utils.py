import streamlit as st

CATEGORY_COLORS = {
    "spam": "#64748b",
    "billing_issue": "#fbbf24",
    "technical_support": "#38bdf8",
    "meeting_request": "#a78bfa",
    "sales_inquiry": "#f472b6",
    "urgent_escalation": "#f87171",
    "general_info": "#94a3b8",
    "internal": "#34d399",
}

CATEGORY_ICONS = {
    "spam": "🗑️",
    "billing_issue": "💰",
    "technical_support": "🔧",
    "meeting_request": "📅",
    "sales_inquiry": "💼",
    "urgent_escalation": "🚨",
    "general_info": "ℹ️",
    "internal": "🏢",
}

TEAM_ICONS = {
    "engineering": "⚙️",
    "finance": "🏦",
    "sales": "📊",
    "support": "🎧",
}

ACTION_ICONS = {
    "read_thread": "📖",
    "classify": "🏷️",
    "set_priority": "⚡",
    "route": "📤",
    "archive": "📥",
    "escalate": "🚀",
    "skip": "⏭️",
}


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


def get_category_badge(cat: str) -> str:
    color = CATEGORY_COLORS.get(cat, "#94a3b8")
    icon = CATEGORY_ICONS.get(cat, "❓")
    label = cat.replace("_", " ").title()
    return f'<span class="badge" style="background:rgba({_hex_to_rgb(color)},0.15);color:{color}">{icon} {label}</span>'


def get_priority_badge(pri: str) -> str:
    colors = {"high": "#f87171", "medium": "#fbbf24", "low": "#34d399"}
    icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    c = colors.get(pri, "#94a3b8")
    return f'<span class="badge" style="background:rgba({_hex_to_rgb(c)},0.15);color:{c}">{icons.get(pri,"")} {pri.upper()}</span>'


def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: #1e293b;
        --border: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1400px !important;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-green { background: rgba(52,211,153,0.15); color: #34d399; }
    .badge-red { background: rgba(248,113,113,0.15); color: #f87171; }
    .badge-amber { background: rgba(251,191,36,0.15); color: #fbbf24; }
    .badge-blue { background: rgba(56,189,248,0.15); color: #38bdf8; }
    .badge-purple { background: rgba(167,139,250,0.15); color: #a78bfa; }

    .email-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
        transition: border-color 0.2s;
    }
    .email-card:hover { border-color: #38bdf8; }
    .email-subject {
        font-weight: 600;
        color: #f1f5f9;
        font-size: 0.9rem;
    }
    .email-sender {
        font-size: 0.78rem;
        color: #64748b;
    }
    .email-body {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 0.4rem;
        line-height: 1.5;
    }

    .action-entry {
        background: #0f172a;
        border-left: 3px solid #38bdf8;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.45rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
    }
    .action-entry.negative { border-left-color: #f87171; }
    .action-entry.positive { border-left-color: #34d399; }

    .reward-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.82rem;
    }
    .reward-table th {
        text-align: left;
        padding: 0.5rem 0.7rem;
        color: #94a3b8;
        font-weight: 500;
        border-bottom: 1px solid #334155;
    }
    .reward-table td {
        padding: 0.4rem 0.7rem;
        color: #cbd5e1;
        border-bottom: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }

    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .workflow-step {
        display: inline-flex;
        align-items: center;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
    }
    .ws-read { background: rgba(56,189,248,0.15); color: #38bdf8; }
    .ws-classify { background: rgba(167,139,250,0.15); color: #a78bfa; }
    .ws-priority { background: rgba(251,191,36,0.15); color: #fbbf24; }
    .ws-terminal { background: rgba(52,211,153,0.15); color: #34d399; }

    .scroll-container {
        max-height: 420px;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    .scroll-container::-webkit-scrollbar { width: 5px; }
    .scroll-container::-webkit-scrollbar-track { background: transparent; }
    .scroll-container::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)


try:
    from inference import (
        SYSTEM_PROMPT,
        _extract_json,
        get_llm_action,
        _decide_next_action_heuristic as heuristic_action
    )
except ImportError:
    # Fallback if inference.py is moved/missing
    SYSTEM_PROMPT = "You are an email triage AI."
    def _extract_json(t): return None
    def heuristic_action(o): return {"action_type": "skip", "email_id": "none"}
    def get_llm_action(c, o, h, v=False): return heuristic_action(o), "Heuristic"

# ── LLM Integration ──────────────────────────────────────────────────────

import os
import json
import re
from typing import Optional, List

_llm_client = None

def get_llm_client():
    global _llm_client
    if _llm_client is None:
        import dotenv
        dotenv.load_dotenv(override=True)
        from openai import OpenAI
        api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        api_base = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
        if not api_key:
            raise ValueError("No API_KEY or HF_TOKEN found in .env")
        _llm_client = OpenAI(base_url=api_base, api_key=api_key)
    return _llm_client

def llm_action(obs: dict, env_dict: dict, action_history: List[str] = None) -> dict:
    if action_history is None:
        action_history = []
    
    llm = get_llm_client()
    
    # We use the exactly same function from inference.py to get the action
    # Note: inference.get_llm_action takes (client, obs, history, verbose)
    # The 'obs' inside inference.py prompt builder uses:
    # obs['current_step'], obs['max_steps'], obs['processed_count'], obs['sla_violations']
    # Our 'obs' dict from the dashboard already contains these fields.
    
    action, method = get_llm_action(llm, obs, action_history)
    return action

