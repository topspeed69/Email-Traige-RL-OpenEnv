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

class EmailDisposition(str, Enum):
    """Terminal disposition for an email"""
    ROUTED = "routed"
    ARCHIVED = "archived"
    ESCALATED = "escalated"

class Email(BaseModel):
    """Single email in the system"""
    id: str
    subject: str
    sender: str
    body: str
    arrival_step: int
    thread_id: Optional[str] = None
    parent_email_id: Optional[str] = None
    depends_on: List[str] = []  # List of email IDs this one depends on
    thread_context: Optional[str] = None  # Hidden context revealed after read_thread
    # Ground truth (hidden from agent)
    true_category: EmailCategory
    true_priority: Literal["high", "medium", "low"]
    true_team: str

class EmailProgress(BaseModel):
    """Tracks incremental progress on a single email"""
    category: Optional[EmailCategory] = None
    priority: Optional[Literal["high", "medium", "low"]] = None
    disposition: Optional[EmailDisposition] = None
    team: Optional[Literal["engineering", "finance", "sales", "support"]] = None
    classified_step: Optional[int] = None
    priority_step: Optional[int] = None
    disposition_step: Optional[int] = None

    @property
    def is_done(self) -> bool:
        """email_done = category_set AND priority_set AND (routed OR archived OR escalated)"""
        return (
            self.category is not None
            and self.priority is not None
            and self.disposition is not None
        )

class Observation(BaseModel):
    """What the agent observes each step"""
    inbox: List[Dict]  # Unprocessed emails (id, subject, sender, body, thread_id, thread_read)
    in_progress: List[Dict]  # Emails with partial progress (not yet done)
    processed_count: int  # Fully done emails
    current_step: int
    max_steps: int
    total_cost: float
    sla_violations: int
    threads_read: List[str] = []  # Thread IDs that have been read
    last_action_error: Optional[str] = None

class Action(BaseModel):
    """Agent's action — one atomic operation per step"""
    action_type: str
    email_id: str
    
    # Kept as free-form strings to capture LLM hallucinations natively
    # Only used when action_type == "classify"
    category: Optional[str] = None
    # Only used when action_type == "set_priority"
    priority: Optional[str] = None
    # Only used when action_type == "route"
    team: Optional[str] = None

class Reward(BaseModel):
    """Reward breakdown"""
    total: float
    category_correct: float
    priority_correct: float
    routing_correct: float
    sla_penalty: float
    action_cost: float
    dependency_violation: float
    completion_bonus: float = 0.0
