import time
from collections import defaultdict
import streamlit as st
import plotly.graph_objects as go

from server.tasks import TASK_CONFIGS
from ui.utils import (
    CATEGORY_COLORS, ACTION_ICONS, 
    get_category_badge, heuristic_action
)
from ui.state import reset_env, execute_step

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;margin-bottom:1.2rem;">
            <div style="font-size:2.5rem;">📧</div>
            <div style="font-size:1.3rem;font-weight:800;color:#f1f5f9;margin-top:0.3rem;">Email Triage RL</div>
            <div style="font-size:0.78rem;color:#64748b;margin-top:0.2rem;">Interactive Environment Viewer</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div class="section-title">🎯 Task Configuration</div>', unsafe_allow_html=True)
        task_options = ["easy", "medium", "hard", "all"]
        task = st.selectbox(
            "Difficulty Level",
            task_options,
            index=task_options.index(st.session_state.task_id),
            key="task_selector",
            format_func=lambda x: {
                "easy": "🟢 Easy — 15 emails, 50 steps",
                "medium": "🟡 Medium — 25 emails, 100 steps",
                "hard": "🔴 Hard — 30 emails, 130 steps",
                "all": "🔥 All (Sequential Run)"
            }[x],
        )

        if task != st.session_state.task_id:
            st.session_state.task_id = task
            reset_env()
            st.rerun()

        cfg = TASK_CONFIGS[st.session_state.current_run_task]
        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:0.7rem 0.9rem;font-size:0.8rem;color:#94a3b8;margin-top:0.5rem;">
            <div><strong style="color:#e2e8f0;">{cfg['name']}</strong></div>
            <div style="margin-top:0.3rem;">📬 {cfg['num_emails']} emails &nbsp;|&nbsp; 🔄 {cfg['max_steps']} max steps</div>
            <div>{'⏰ SLA Penalties' if cfg.get('has_sla') else '✖ No SLA'} &nbsp;|&nbsp; {'🔗 Dependencies' if cfg.get('has_dependencies') else '✖ No Deps'}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div class="section-title">🤖 Agent Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Select agent",
            ["heuristic", "llm_agent", "manual"],
            format_func=lambda x: {
                "heuristic": "🧠 Heuristic (Auto)",
                "llm_agent": "🤖 Live LLM Agent",
                "manual": "🎮 Manual Control",
            }[x],
            key="mode_radio",
            horizontal=False,
        )
        st.session_state.mode = mode

        if mode in ["heuristic", "llm_agent"]:
            st.session_state.auto_speed = st.slider(
                "Step delay (sec)", 0.0, 1.0, st.session_state.auto_speed, 0.05,
                help="Delay between auto-steps for visualization"
            )
        
        if mode == "llm_agent":
            import os
            import dotenv
            dotenv.load_dotenv(override=True)
            mod = os.getenv("MODEL_NAME", "mistralai/mistral-7b-instruct-v0.2")
            st.markdown(f'<div style="font-size:0.75rem;color:#64748b;margin-top:-0.5rem;">Model: <strong style="color:#94a3b8;">{mod.split("/")[-1]}</strong></div>', unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div class="section-title">🎛️ Controls</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Reset", use_container_width=True, type="secondary"):
                reset_env()
                st.rerun()
        with col_b:
            if st.session_state.env is None:
                if st.button("▶️ Start", use_container_width=True, type="primary"):
                    reset_env()
                    st.rerun()

        st.markdown("---")

        st.markdown('<div class="section-title">📐 Workflow Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem;color:#94a3b8;line-height:1.7;">
            <span class="workflow-step ws-read">📖 read_thread</span>
            <span style="color:#475569">→</span>
            <span class="workflow-step ws-classify">🏷️ classify</span>
            <span style="color:#475569">→</span>
            <span class="workflow-step ws-priority">⚡ set_priority</span>
            <span style="color:#475569">→</span>
            <span class="workflow-step ws-terminal">📤 route / 📥 archive / 🚀 escalate</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        with st.expander("📊 Reward Reference", expanded=False):
            st.markdown("""
            | Component | Value |
            |-----------|-------|
            | Category ✅ | +0.5 to +3.0 |
            | Category ❌ | -1.0 |
            | Priority ✅ | +1.0 |
            | Priority ❌ | -0.3 |
            | Routing ✅ | +1.0 |
            | Routing ❌ | -0.5 |
            | SLA Violation | -2.0 |
            | Dep. Violation | -1.0 |
            | Completion ⚡ | +0.25 / +0.5 |
            """)

def render_landing_page():
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">📧</div>
        <h1 style="font-size:2.2rem;font-weight:800;color:#f1f5f9;margin-bottom:0.5rem;">
            Email Triage RL Environment
        </h1>
        <p style="font-size:1.1rem;color:#94a3b8;max-width:650px;margin:0 auto 2rem;">
            An interactive visualization of the reinforcement learning environment for email inbox management.
            Watch a heuristic agent process emails through classification, prioritization, and routing —
            or take manual control yourself.
        </p>
        <div style="display:flex;justify-content:center;gap:1.5rem;margin-bottom:2.5rem;flex-wrap:wrap;">
            <div class="metric-card" style="min-width:140px;">
                <div class="metric-value" style="color:#34d399;">8</div>
                <div class="metric-label">Categories</div>
            </div>
            <div class="metric-card" style="min-width:140px;">
                <div class="metric-value" style="color:#38bdf8;">4</div>
                <div class="metric-label">Teams</div>
            </div>
            <div class="metric-card" style="min-width:140px;">
                <div class="metric-value" style="color:#fbbf24;">3</div>
                <div class="metric-label">Difficulty Levels</div>
            </div>
            <div class="metric-card" style="min-width:140px;">
                <div class="metric-value" style="color:#a78bfa;">6</div>
                <div class="metric-label">Action Types</div>
            </div>
        </div>
        <p style="color:#64748b;font-size:0.85rem;">← Select a task and click <strong>Start</strong> in the sidebar to begin</p>
    </div>
    """, unsafe_allow_html=True)

def render_top_metrics_bar(env, obs):
    status_color = "red" if st.session_state.episode_done else "green"
    status_text = "COMPLETED" if st.session_state.episode_done else "RUNNING"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem;">
        <span style="font-size:1.4rem;">📧</span>
        <span style="font-size:1.2rem;font-weight:700;color:#f1f5f9;">Email Triage Dashboard</span>
        <span class="badge badge-blue" style="margin-left:0.5rem;">Step {env['current_step']}/{env['max_steps']}</span>
        <span class="badge badge-{status_color}">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#38bdf8;">{len(obs.inbox)}</div>
            <div class="metric-label">Inbox</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#fbbf24;">{len(obs.in_progress)}</div>
            <div class="metric-label">In Progress</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#34d399;">{obs.processed_count}</div>
            <div class="metric-label">Processed</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        sla_color = "#34d399" if obs.sla_violations == 0 else "#f87171"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{sla_color};">{obs.sla_violations}</div>
            <div class="metric-label">SLA Violations</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        cum_rwd = st.session_state.cumulative_rewards[-1] if st.session_state.cumulative_rewards else 0.0
        rwd_color = "#34d399" if cum_rwd >= 0 else "#f87171"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{rwd_color};">{cum_rwd:.1f}</div>
            <div class="metric-label">Total Reward</div>
        </div>""", unsafe_allow_html=True)
    with k6:
        if st.session_state.final_score is not None:
            sc = st.session_state.final_score
            sc_color = "#34d399" if sc >= 0.7 else ("#fbbf24" if sc >= 0.4 else "#f87171")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{sc_color};">{sc:.2f}</div>
                <div class="metric-label">Final Score</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#475569;">—</div>
                <div class="metric-label">Final Score</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

def render_inbox_column(col, obs):
    with col:
        st.markdown('<div class="section-title">📬 Inbox</div>', unsafe_allow_html=True)

        if obs.inbox:
            for em in obs.inbox[:8]:
                thread_tag = ""
                if em.get("thread_id"):
                    tread_status = "READ" if em.get("thread_read") else "UNREAD"
                    thread_tag = f' <span class="badge badge-{"green" if tread_status=="READ" else "amber"}" style="font-size:0.65rem;">🔗 {tread_status}</span>'
                dep_tag = ""
                if em.get("depends_on"):
                    dep_tag = f' <span class="badge badge-purple" style="font-size:0.65rem;">🔗 Depends</span>'

                st.markdown(f"""
                <div class="email-card">
                    <div class="email-subject">{em['subject'][:60]}{thread_tag}{dep_tag}</div>
                    <div class="email-sender">From: {em['sender']}</div>
                    <div class="email-body">{(em.get('body') or '')[:120]}{'...' if len(em.get('body',''))>120 else ''}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#475569;font-size:0.85rem;text-align:center;padding:1rem;">📭 Inbox empty</div>', unsafe_allow_html=True)

        # In-Progress
        if obs.in_progress:
            st.markdown(f'<div class="section-title" style="margin-top:1rem;">🔄 In Progress ({len(obs.in_progress)})</div>', unsafe_allow_html=True)
            for ep in obs.in_progress[:5]:
                cat_badge = get_category_badge(ep["category_set"]) if ep.get("category_set") else '<span class="badge badge-amber" style="font-size:0.65rem;">Category: Pending</span>'
                pri_status = "✅ Set" if ep.get("priority_set") else "⏳ Pending"
                st.markdown(f"""
                <div class="email-card" style="border-left:3px solid #fbbf24;">
                    <div class="email-subject">{ep['subject'][:55]}</div>
                    <div style="margin-top:0.3rem;">{cat_badge}</div>
                    <div style="font-size:0.75rem;color:#64748b;margin-top:0.2rem;">Priority: {pri_status}</div>
                </div>
                """, unsafe_allow_html=True)

def render_charts_and_accuracy(col, env):
    with col:
        tab_charts, tab_breakdown, tab_emails_ground = st.tabs(["📈 Live Charts", "🎯 Accuracy", "📋 Ground Truth"])

        with tab_charts:
            if st.session_state.reward_history:
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    y=st.session_state.cumulative_rewards,
                    mode='lines+markers',
                    line=dict(color='#38bdf8', width=2.5),
                    marker=dict(size=4, color='#38bdf8'),
                    fill='tozeroy',
                    fillcolor='rgba(56,189,248,0.08)',
                    name='Cumulative Reward',
                ))
                fig_cum.update_layout(
                    title=dict(text="Cumulative Reward", font=dict(size=14, color='#e2e8f0')),
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=220,
                    margin=dict(l=40, r=20, t=40, b=30),
                    xaxis=dict(title="Step", gridcolor='#1e293b', showline=False),
                    yaxis=dict(gridcolor='#1e293b', showline=False),
                    showlegend=False,
                )
                fig_cum.add_hline(y=0, line_dash="dash", line_color="#475569", opacity=0.5)
                st.plotly_chart(fig_cum, use_container_width=True)

                colors = ['#34d399' if r >= 0 else '#f87171' for r in st.session_state.reward_history]
                fig_step = go.Figure()
                fig_step.add_trace(go.Bar(
                    y=st.session_state.reward_history,
                    marker_color=colors,
                    opacity=0.85,
                    name="Step Reward",
                ))
                fig_step.update_layout(
                    title=dict(text="Per-Step Rewards", font=dict(size=14, color='#e2e8f0')),
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=200,
                    margin=dict(l=40, r=20, t=40, b=30),
                    xaxis=dict(title="Step", gridcolor='#1e293b'),
                    yaxis=dict(gridcolor='#1e293b'),
                    showlegend=False,
                )
                fig_step.add_hline(y=0, line_dash="dash", line_color="#475569", opacity=0.5)
                st.plotly_chart(fig_step, use_container_width=True)
            else:
                st.markdown("""
                <div style="text-align:center;padding:3rem;color:#475569;">
                    <div style="font-size:2rem;margin-bottom:0.5rem;">📊</div>
                    Charts will appear after the first step
                </div>
                """, unsafe_allow_html=True)

        with tab_breakdown:
            if st.session_state.action_log:
                def _acc_pct(d):
                    total = d["correct"] + d["incorrect"]
                    return (d["correct"] / total * 100) if total > 0 else 0

                ca = _acc_pct(st.session_state.category_accuracy)
                pa = _acc_pct(st.session_state.priority_accuracy)
                ra = _acc_pct(st.session_state.routing_accuracy)

                g1, g2, g3 = st.columns(3)
                for c_col, label, val, color in [
                    (g1, "Category", ca, "#a78bfa"),
                    (g2, "Priority", pa, "#fbbf24"),
                    (g3, "Routing", ra, "#38bdf8"),
                ]:
                    with c_col:
                        fig_g = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=val,
                            number=dict(suffix="%", font=dict(size=22, color=color)),
                            gauge=dict(
                                axis=dict(range=[0, 100], tickfont=dict(size=10, color='#475569')),
                                bar=dict(color=color),
                                bgcolor='#1e293b',
                                borderwidth=0,
                                steps=[
                                    dict(range=[0, 40], color='rgba(248,113,113,0.1)'),
                                    dict(range=[40, 70], color='rgba(251,191,36,0.1)'),
                                    dict(range=[70, 100], color='rgba(52,211,153,0.1)'),
                                ],
                            ),
                            title=dict(text=label, font=dict(size=12, color='#94a3b8')),
                        ))
                        fig_g.update_layout(
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            height=180,
                            margin=dict(l=20, r=20, t=35, b=10),
                        )
                        st.plotly_chart(fig_g, use_container_width=True)

                if env.get('progress'):
                    cat_counts = defaultdict(int)
                    for eid, prog in env['progress'].items():
                        if prog.get('category'):
                            cat_counts[prog['category']] += 1
                    if cat_counts:
                        cats = list(cat_counts.keys())
                        vals = list(cat_counts.values())
                        cols = [CATEGORY_COLORS.get(c, "#94a3b8") for c in cats]
                        fig_cat = go.Figure(go.Bar(
                            x=[c.replace("_", " ").title() for c in cats],
                            y=vals,
                            marker_color=cols,
                            opacity=0.9,
                        ))
                        fig_cat.update_layout(
                            title=dict(text="Classifications Made", font=dict(size=13, color='#e2e8f0')),
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=200,
                            margin=dict(l=40, r=20, t=40, b=50),
                            xaxis=dict(gridcolor='#1e293b', tickangle=-30),
                            yaxis=dict(gridcolor='#1e293b', title="Count"),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.markdown("""
                <div style="text-align:center;padding:3rem;color:#475569;">
                    <div style="font-size:2rem;margin-bottom:0.5rem;">🎯</div>
                    Accuracy metrics will appear after actions are taken
                </div>
                """, unsafe_allow_html=True)

        with tab_emails_ground:
            st.markdown('<div class="section-title">📋 Email Ground Truth (Hidden from Agent)</div>', unsafe_allow_html=True)
            email_data = []
            for em in env.get('emails', [])[:20]:
                prog = env['progress'].get(em['id'])
                status = "✅ Done" if (prog and prog.get('is_done')) else ("🔄 In Progress" if prog else "📬 Inbox")
                agent_cat = prog.get('category') if (prog and prog.get('category')) else "—"
                agent_pri = prog.get('priority') if (prog and prog.get('priority')) else "—"
                match_status = "✅" if (prog and prog.get('category') and prog.get('category') == em.get('true_category')) else ("❌" if (prog and prog.get('category')) else "—")
                email_data.append({
                    "ID": em['id'],
                    "Subject": em.get('subject', '')[:40],
                    "True Cat.": em.get('true_category'),
                    "Agent Cat.": agent_cat,
                    "Match?": match_status,
                    "True Pri.": em.get('true_priority'),
                    "Agent Pri.": agent_pri,
                    "True Team": em.get('true_team'),
                    "Status": status,
                })
            st.dataframe(email_data, use_container_width=True, height=400)

def render_action_log(col):
    with col:
        st.markdown('<div class="section-title">📝 Action Log</div>', unsafe_allow_html=True)

        if st.session_state.action_log:
            log_html = '<div class="scroll-container">'
            for entry in reversed(st.session_state.action_log[-30:]):
                act = entry["action"]
                rwd = entry["reward"]
                icon = ACTION_ICONS.get(act.get("action_type", ""), "❓")
                cls = entry["reward_color"]

                detail = f"{act.get('action_type','?')}({act.get('email_id','?')})"
                extras = []
                if act.get("category"):
                    extras.append(f"cat={act['category']}")
                if act.get("priority"):
                    extras.append(f"pri={act['priority']}")
                if act.get("team"):
                    extras.append(f"team={act['team']}")
                if extras:
                    detail += f" [{', '.join(extras)}]"

                err_html = f'<div style="color:#f87171;font-size:0.72rem;">⚠ {entry["error"]}</div>' if entry.get("error") else ""

                log_html += f'<div class="action-entry {cls}">'
                log_html += f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                log_html += f'<span>{icon} <strong>Step {entry["step"]}</strong></span>'
                rwd_amt = f"{'+' if rwd['total']>=0 else ''}{rwd['total']:.2f}"
                rwd_col = "#34d399" if rwd['total'] >= 0 else "#f87171"
                log_html += f'<span style="color:{rwd_col};font-weight:600;font-size:0.85rem;">{rwd_amt}</span>'
                log_html += f'</div><div style="color:#cbd5e1;font-size:0.8rem;margin-top:0.15rem;">{detail}</div>'
                log_html += f'{err_html}</div>\n'
                
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:2rem;color:#475569;font-size:0.85rem;">
                Actions will appear here as the agent works
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.action_log:
            last_rwd = st.session_state.action_log[-1]["reward"]
            with st.expander("🔍 Last Reward Breakdown", expanded=False):
                for k, v in last_rwd.items():
                    if k == "total":
                        continue
                    v_str = f"{v:+.2f}" if v != 0 else "0.00"
                    color = "#34d399" if v > 0 else ("#f87171" if v < 0 else "#475569")
                    st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:0.15rem 0;"><span style="color:#94a3b8;">{k.replace("_"," ").title()}</span><span style="color:{color};font-weight:600;">{v_str}</span></div>', unsafe_allow_html=True)

def render_episode_completion(env, obs):
    score = st.session_state.final_score or 0.0
    grade = "A+" if score >= 0.9 else ("A" if score >= 0.8 else ("B" if score >= 0.7 else ("C" if score >= 0.5 else "D")))
    grade_color = "#34d399" if score >= 0.7 else ("#fbbf24" if score >= 0.5 else "#f87171")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;border-radius:14px;padding:1.5rem 2rem;margin-top:1rem;text-align:center;">
        <div style="font-size:1.5rem;font-weight:800;color:#e2e8f0;">🏁 Episode Complete</div>
        <div style="display:flex;justify-content:center;gap:2.5rem;margin-top:1rem;flex-wrap:wrap;">
            <div>
                <div style="font-size:2.2rem;font-weight:800;color:{grade_color};">{score:.3f}</div>
                <div style="font-size:0.78rem;color:#64748b;">FINAL SCORE</div>
            </div>
            <div>
                <div style="font-size:2.2rem;font-weight:800;color:{grade_color};">{grade}</div>
                <div style="font-size:0.78rem;color:#64748b;">GRADE</div>
            </div>
            <div>
                <div style="font-size:2.2rem;font-weight:800;color:#38bdf8;">{env['current_step']}</div>
                <div style="font-size:0.78rem;color:#64748b;">STEPS TAKEN</div>
            </div>
            <div>
                <div style="font-size:2.2rem;font-weight:800;color:#a78bfa;">{obs.processed_count}</div>
                <div style="font-size:0.78rem;color:#64748b;">EMAILS DONE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # All Tasks Chaining Logic
    if st.session_state.task_id == "all":
        current = st.session_state.current_run_task
        nxt = None
        if current == "easy": nxt = "medium"
        elif current == "medium": nxt = "hard"
        
        if nxt:
            st.markdown(f"""
            <div style="text-align:center;margin-top:1.5rem;color:#94a3b8;font-size:0.9rem;">
                Running in "All" mode. Ready to proceed to <strong>{nxt.upper()}</strong> class.
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([1,1,1])
            with c2:
                if st.button(f"🚀 Proceed to {nxt.title()}", type="primary", use_container_width=True):
                    reset_env(target_task=nxt)
                    if st.session_state.mode != "manual":
                        st.session_state.is_running = True
                    st.rerun()

def render_manual_controls(obs):
    st.markdown('<div class="section-title">🎮 Manual Action</div>', unsafe_allow_html=True)

    available_ids = [e["id"] for e in obs.inbox] + [e["id"] for e in obs.in_progress]
    if not available_ids:
        available_ids = ["none"]

    m1, m2 = st.columns(2)
    with m1:
        action_type = st.selectbox(
            "Action Type",
            ["classify", "set_priority", "route", "archive", "escalate", "read_thread", "skip"],
        )
    with m2:
        email_id = st.selectbox("Email ID", available_ids)

    m3, m4, m5 = st.columns(3)
    with m3:
        category = st.selectbox(
            "Category (for classify)",
            [None, "spam", "billing_issue", "technical_support", "meeting_request",
             "sales_inquiry", "urgent_escalation", "general_info", "internal"],
        )
    with m4:
        priority = st.selectbox("Priority (for set_priority)", [None, "high", "medium", "low"])
    with m5:
        team = st.selectbox("Team (for route)", [None, "engineering", "finance", "sales", "support"])

    if st.button("▶️ Execute Action", type="primary", use_container_width=True):
        act = {"action_type": action_type, "email_id": email_id}
        if category: act["category"] = category
        if priority: act["priority"] = priority
        if team: act["team"] = team
        execute_step(act)
        st.rerun()

from ui.utils import llm_action

def run_agent_loop(env, obs_dict):
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("⏯️ Step Once", use_container_width=True):
            if st.session_state.mode == "llm_agent":
                try:
                    with st.spinner("🤖 Agent thinking..."):
                        hist = [e["action"].get("action_type","") for e in st.session_state.action_log[-3:]]
                        action = llm_action(obs_dict, env, hist)
                except Exception as e:
                    st.error(f"LLM Error: {e}")
                    return
            else:
                action = heuristic_action(obs_dict)
            execute_step(action)
            st.rerun()
    with c2:
        if st.button("🚀 Run All", use_container_width=True, type="primary"):
            st.session_state.is_running = True

    if st.session_state.is_running and not st.session_state.episode_done:
        # Instead of a blocking while loop, we execute precisely one step 
        # and re-run Streamlit to allow the GUI to update incrementally.
        obs_d = st.session_state.obs.model_dump()
        
        if st.session_state.mode == "llm_agent":
            hist = [e["action"].get("action_type","") for e in st.session_state.action_log[-3:]]
            try:
                with st.spinner("🤖 Agent thinking..."):
                    action = llm_action(obs_d, env, hist)
            except Exception as e:
                st.error(f"LLM crashed: {e}")
                st.session_state.is_running = False
                st.rerun(scope="fragment")
                return
        else:
            action = heuristic_action(obs_d)
            
        execute_step(action)
        
        # Add visual delay for observation (as tuned by user)
        if st.session_state.auto_speed > 0:
            time.sleep(st.session_state.auto_speed)
            
        if st.session_state.episode_done:
            st.session_state.is_running = False
            
        st.rerun(scope="fragment")
