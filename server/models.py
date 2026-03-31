from pydantic import BaseModel
from typing import List, Dict, Optional, Literal
from enum import Enum

class EmailCategory(str, Enum):
    SPAM = "spam"
    BILLING_ISSUE = "billing_issue"
    TECHNICAL_SUPPORT = "technical_support"
    MEETING_REQUEST = "meeting_request"
    SALES_INQUIRY = "sales_inquiry"
    URGENT_ESCALATION = "urgent_escalation"
    GENERAL_INFO = "general_info"
    INTERNAL = "internal"

class Email(BaseModel):
    """Single email in the system"""
    id: str
    subject: str
    sender: str
    body: str
    arrival_step: int
    thread_id: Optional[str] = None
    parent_email_id: Optional[str] = None
    # Ground truth (hidden from agent)
    true_category: EmailCategory
    true_priority: Literal["high", "medium", "low"]
    true_team: str

class Observation(BaseModel):
    """What the agent observes each step"""
    inbox: List[Dict]  # Unprocessed emails (id, subject, sender, body, thread_id)
    processed_count: int
    current_step: int
    max_steps: int
    total_cost: float
    sla_violations: int
    last_action_error: Optional[str] = None

class Action(BaseModel):
    """Agent's action"""
    action_type: Literal[
        "classify",
        "route", 
        "archive",
        "escalate",
        "read_thread",
        "skip"
    ]
    email_id: str
    category: Optional[EmailCategory] = None
    priority: Optional[Literal["high", "medium", "low"]] = None
    team: Optional[Literal["engineering", "finance", "sales", "support"]] = None

class Reward(BaseModel):
    """Reward breakdown"""
    total: float
    category_correct: float
    priority_correct: float
    routing_correct: float
    sla_penalty: float
    action_cost: float
    dependency_violation: float
