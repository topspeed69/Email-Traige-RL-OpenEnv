# EmailTriageEnv

A realistic inbox management system compliant with the OpenEnv architecture.

## Overview
This environment tasks an AI agent with processing a realistic email inbox. In the real world, human triage agents handle incoming streams of email and categorize them, assign priority based on SLAs, and route them to downstream teams. This environment directly models these workflows, serving as a functional evaluation gate for language model agents managing structured workflows with consequences (SLAs, wrong routing costs).

The agent is responsible for:
- Classifying emails into 8 strict categories
- Assigning priorities (high/medium/low)
- Routing to correct teams
- Handling dependencies (meaning reading email threads instead of single messages when necessary)
- Minimizing cost (action costs + decay penalties for late action)
- Meeting SLA requirements for time-sensitive categories

## Actions and Observation Spaces

### Observation Space
The agent observes the environment via the following structured JSON schema:
- `inbox` (array): List of unprocessed email objects (contains id, subject, sender, body preview, thread_id, arrival_step). The true category/priority/team labels are hidden from the agent.
- `processed_count` (int): Number of emails already processed.
- `current_step` (int): The current time step.
- `max_steps` (int): The maximum steps allowed for the task before truncation.
- `total_cost` (number): Current accumulated cost/penalty.
- `sla_violations` (int): Count of SLAs missed so far.
- `last_action_error` (string, optional): Feedback from the environment if the last action was invalid.

### Action Space
The agent produces an Action containing the following fields:
- `action_type` (string): The intended action. Must be one of `[classify, route, archive, escalate, read_thread, skip]`.
- `email_id` (string): The ID of the email to operate on.
- `category` (string, optional): One of the 8 predefined categories (`spam`, `billing_issue`, `technical_support`, `meeting_request`, `sales_inquiry`, `urgent_escalation`, `general_info`, `internal`).
- `priority` (string, optional): The priority assignment (`high`, `medium`, `low`).
- `team` (string, optional): Target team routing (`engineering`, `finance`, `sales`, `support`).

## Reward and Penalty Calculations
The environment provides dense, step-by-step rewards based on multiple components. The total reward for taking an action is the sum of the following factors:

### 1. Accuracy Bonuses & Penalties
- **Category Match:** Rewards vary by the true category's importance.
  - `urgent_escalation`: +3.0
  - `billing_issue`: +2.5
  - `technical_support`: +2.0
  - `meeting_request`: +1.5
  - `sales_inquiry`: +1.0
  - `general_info`: +0.8
  - `internal`: +0.8
  - `spam`: +0.5
  - *Incorrect Category Penalty:* -1.0
- **Priority Match:** Correct (+1.0) / Incorrect (-0.3)
- **Routing Match:** Correct (+1.0) / Incorrect (-0.5)

### 2. SLA & Dependency Penalties
- **SLA Penalty (-2.0):** Applied if an email isn't processed within its Service Level Agreement timeframe:
  - `urgent_escalation`: SLA limit is 10 steps.
  - `billing_issue`: SLA limit is 15 steps.
- **Dependency Violation (-1.0):** Applied if the agent classifies an email containing a `thread_id` without checking the thread context first.

### 3. Action Costs
Every API/Tool action inherently costs resources (simulating API latency or compute cost):
- `classify`: -0.02
- `route`: -0.02
- `archive`: -0.01
- `read_thread`: -0.01
- `escalate`: -0.50
- `skip`: 0.00
- *Invalid Actions:* default -0.05

### 4. Wait Decay (Time Penalty)
Every step that urgent emails remain sitting unprocessed in the inbox applies a cumulative penalty:
- `urgent_escalation`: -0.15 per step
- `billing_issue`: -0.10 per step

## Tasks and Difficulty
We offer 3 standardized tasks evaluated by objective agent-graders:

1. **easy (Basic Email Classification)**: Classifies 15 emails strictly into 4 basic categories. No SLA limits. Expected to be solvable (0.9+ score) by smaller models.
2. **medium (Priority-Aware Triage)**: Triages 25 emails over 40 steps, across all categories. Introduces SLAs (requiring the agent to prioritize urgent items immediately). Grader punishes SLA misses. Expected to challenge standard 7B models.
3. **hard (Full Inbox Management)**: The full pipeline with 30 emails and 50 steps. Evaluates all components: routing, category, SLA, priorities, and dependency threading. Demands strong planning capabilities and serves as a benchmark for frontier models.

## Setup and Running

Install dependencies using `uv`:
```bash
uv lock
uv sync
```

### Running the API
The environment server can be launched locally for interactions:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Running the Inference Baseline
The repository includes a baseline inference script utilizing the OpenAI API specification. To execute it:
```bash
export API_BASE_URL="https://api.openai.com/v1"  # Or another compatible proxy
export MODEL_NAME="gpt-4o"
export OPENAI_API_KEY="your_openai_api_key_here"
# OR alternatively:
# export HF_TOKEN="your_hf_token_here"

python inference.py
```

### Testing
```bash
pytest
```

## Baseline Scores
Testing the `inference.py` script via a typical frontier model yields reproducible scores reflecting the increasing difficulty:

- **easy**: ~0.95
- **medium**: ~0.82
- **hard**: ~0.65 
