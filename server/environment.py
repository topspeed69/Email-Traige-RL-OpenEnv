from typing import Tuple, Dict, Any, Optional, List
import random
from .models import Observation, Action, Reward, Email, EmailCategory
from .tasks import TASK_CONFIGS

class EmailTriageEnv:
    """OpenEnv-compliant email triage environment"""
    
    def __init__(self):
        self.current_task = None
        self.emails = []
        self.processed = {}
        self.current_step = 0
        self.max_steps = 0
        self.cumulative_reward = 0.0
        self.sla_violations = 0
        self.dependency_violations = 0
        
    def reset(self, task_id: str) -> Observation:
        """
        Reset environment for a specific task.
        
        Args:
            task_id: One of ["easy", "medium", "hard"]
            
        Returns:
            Initial observation
        """
        task_config = TASK_CONFIGS[task_id]
        
        # Reset state
        self.current_task = task_id
        self.emails = self._generate_emails(task_id)
        self.processed = {}
        self.current_step = 0
        self.max_steps = task_config["max_steps"]
        self.cumulative_reward = 0.0
        self.sla_violations = 0
        self.dependency_violations = 0
        
        return self._get_observation()
    
    def state(self) -> Observation:
        """
        Get the current observation state.
        
        Returns:
            Current static observation
        """
        return self._get_observation()
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, bool, Dict[str, Any]]:
        """
        Execute one step.
        
        Args:
            action: Agent's action
            
        Returns:
            observation: Current state observation
            reward: Reward breakdown
            done: Episode terminated naturally
            truncated: Episode hit max steps
            info: Additional info dict
        """
        # Validate action
        email = self._get_email(action.email_id)
        error_msg = None
        
        if not email:
            error_msg = f"Invalid email_id: {action.email_id}"
            reward = Reward(total=-0.5, category_correct=0, priority_correct=0, 
                          routing_correct=0, sla_penalty=0, action_cost=-0.5,
                          dependency_violation=0)
        else:
            # Calculate reward
            reward = self._calculate_reward(action, email)
            
            # Update state
            if action.action_type in ["classify", "route", "archive"]:
                self._process_email(action, email)
        
        # Increment step
        self.current_step += 1
        self.cumulative_reward += reward.total
        
        # Check termination
        done = self._is_naturally_done()
        truncated = self.current_step >= self.max_steps
        
        # Build info
        info = {
            "step": self.current_step,
            "cumulative_reward": self.cumulative_reward,
            "sla_violations": self.sla_violations,
            "dependency_violations": self.dependency_violations
        }
        
        # Get observation
        obs = self._get_observation(error_msg)
        
        return obs, reward, done, truncated, info
    
    def _calculate_reward(self, action: Action, email: Email) -> Reward:
        """Calculate multi-component reward"""
        
        category_correct = 0.0
        priority_correct = 0.0
        routing_correct = 0.0
        sla_penalty = 0.0
        dependency_violation = 0.0
        
        # 1. Category accuracy
        if action.category:
            CATEGORY_WEIGHTS = {
                EmailCategory.URGENT_ESCALATION: 3.0,
                EmailCategory.BILLING_ISSUE: 2.5,
                EmailCategory.TECHNICAL_SUPPORT: 2.0,
                EmailCategory.MEETING_REQUEST: 1.5,
                EmailCategory.SALES_INQUIRY: 1.0,
                EmailCategory.GENERAL_INFO: 0.8,
                EmailCategory.SPAM: 0.5,
                EmailCategory.INTERNAL: 0.8
            }
            
            if action.category == email.true_category:
                category_correct = CATEGORY_WEIGHTS[email.true_category]
            else:
                category_correct = -1.0
        
        # 2. Priority correctness
        if action.priority:
            if action.priority == email.true_priority:
                priority_correct = 1.0
            else:
                priority_correct = -0.3
        
        # 3. Routing correctness
        if action.team:
            if action.team == email.true_team:
                routing_correct = 1.0
            else:
                routing_correct = -0.5
        
        # 4. SLA penalty
        if email.true_category in [EmailCategory.BILLING_ISSUE, EmailCategory.URGENT_ESCALATION]:
            steps_elapsed = self.current_step - email.arrival_step
            sla_threshold = 10 if email.true_category == EmailCategory.URGENT_ESCALATION else 15
            
            if steps_elapsed > sla_threshold:
                sla_penalty = -2.0
                self.sla_violations += 1
        
        # 5. Dependency violation
        if action.action_type == "classify" and email.thread_id:
            if not self._thread_read(email.thread_id):
                dependency_violation = -1.0
                self.dependency_violations += 1
        
        # 6. Action cost
        ACTION_COSTS = {
            "classify": 0.02,
            "route": 0.02,
            "archive": 0.01,
            "escalate": 0.5,
            "read_thread": 0.01,
            "skip": 0.0
        }
        action_cost = -ACTION_COSTS.get(action.action_type, 0.05)
        
        # 7. Decay for unprocessed urgent emails
        decay = self._calculate_decay()
        
        total = (category_correct + priority_correct + routing_correct + 
                sla_penalty + action_cost + dependency_violation + decay)
        
        return Reward(
            total=total,
            category_correct=category_correct,
            priority_correct=priority_correct,
            routing_correct=routing_correct,
            sla_penalty=sla_penalty,
            action_cost=action_cost,
            dependency_violation=dependency_violation
        )
    
    def _calculate_decay(self) -> float:
        """Decay penalty for unprocessed urgent emails"""
        decay = 0.0
        for email in self.emails:
            if email.id not in self.processed:
                if email.true_category == EmailCategory.URGENT_ESCALATION:
                    decay -= 0.15
                elif email.true_category == EmailCategory.BILLING_ISSUE:
                    decay -= 0.1
        return decay
    
    def _get_observation(self, error_msg: Optional[str] = None) -> Observation:
        """Build observation (hide ground truth)"""
        inbox = []
        for email in self.emails:
            if email.id not in self.processed:
                inbox.append({
                    "id": email.id,
                    "subject": email.subject,
                    "sender": email.sender,
                    "body": email.body,
                    "thread_id": email.thread_id,
                    "arrival_step": email.arrival_step
                })
        
        return Observation(
            inbox=inbox,
            processed_count=len(self.processed),
            current_step=self.current_step,
            max_steps=self.max_steps,
            total_cost=abs(self.cumulative_reward) if self.cumulative_reward < 0 else 0,
            sla_violations=self.sla_violations,
            last_action_error=error_msg
        )
    
    def _is_naturally_done(self) -> bool:
        """Check if episode completed naturally"""
        # All emails processed
        if len(self.processed) == len(self.emails):
            return True
        # Too many violations (early termination)
        if self.sla_violations > 10:
            return True
        return False
    
    def _process_email(self, action: Action, email: Email):
        """Mark email as processed"""
        self.processed[email.id] = {
            "category": action.category,
            "priority": action.priority,
            "team": action.team,
            "step": self.current_step
        }
    
    def _get_email(self, email_id: str) -> Optional[Email]:
        """Find email by ID"""
        for email in self.emails:
            if email.id == email_id:
                return email
        return None
    
    def _thread_read(self, thread_id: str) -> bool:
        """Check if thread was read"""
        # Simplified: check if we've processed any email in this thread
        for email in self.emails:
            if email.thread_id == thread_id and email.id in self.processed:
                return True
        return False
    
    def _generate_emails(self, task_id: str) -> List[Email]:
        """Load synthetic emails for task"""
        from .email_loader import EmailLoader
        loader = EmailLoader()
        return loader.load(task_id)
