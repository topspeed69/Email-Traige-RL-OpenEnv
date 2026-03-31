from typing import Tuple, Dict, Any, Optional, List
from .models import (
    Observation, Action, Reward, Email, EmailCategory,
    EmailProgress, EmailDisposition
)
from .tasks import TASK_CONFIGS

class EmailTriageEnv:
    """OpenEnv-compliant email triage environment with multi-step workflows"""
    
    def __init__(self):
        self.current_task = None
        self.emails: List[Email] = []
        self.progress: Dict[str, EmailProgress] = {}  # per-email incremental state
        self.threads_read: set = set()  # thread_ids that have been read
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
        self.progress = {}
        self.threads_read = set()
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
            action: Agent's action (one atomic operation)
            
        Returns:
            observation, reward, done, truncated, info
        """
        # Validate email exists
        email = self._get_email(action.email_id)
        error_msg = None
        
        if not email and action.action_type != "skip":
            error_msg = f"Invalid email_id: {action.email_id}"
            reward = Reward(
                total=-0.5, category_correct=0, priority_correct=0,
                routing_correct=0, sla_penalty=0, action_cost=-0.5,
                dependency_violation=0, completion_bonus=0
            )
        elif action.action_type == "skip":
            reward = Reward(
                total=0.0, category_correct=0, priority_correct=0,
                routing_correct=0, sla_penalty=0, action_cost=0,
                dependency_violation=0, completion_bonus=0
            )
        else:
            # Validate & execute the action, compute reward
            reward, error_msg = self._execute_action(action, email)
        
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
            "dependency_violations": self.dependency_violations,
            "emails_done": sum(1 for p in self.progress.values() if p.is_done),
            "emails_in_progress": sum(1 for p in self.progress.values() if not p.is_done),
        }
        
        # Get observation
        obs = self._get_observation(error_msg)
        
        return obs, reward, done, truncated, info
    
    # ─── Action Execution ───────────────────────────────────────────────
    
    def _execute_action(self, action: Action, email: Email) -> Tuple[Reward, Optional[str]]:
        """Route to the correct handler, return (reward, error_msg)."""
        
        handlers = {
            "read_thread": self._handle_read_thread,
            "classify": self._handle_classify,
            "set_priority": self._handle_set_priority,
            "route": self._handle_route,
            "archive": self._handle_archive,
            "escalate": self._handle_escalate,
        }
        
        handler = handlers.get(action.action_type)
        if not handler:
            return (
                Reward(total=-0.5, category_correct=0, priority_correct=0,
                       routing_correct=0, sla_penalty=0, action_cost=-0.5,
                       dependency_violation=0, completion_bonus=0),
                f"Unknown action_type: {action.action_type}"
            )
        
        return handler(action, email)
    
    def _handle_read_thread(self, action: Action, email: Email) -> Tuple[Reward, Optional[str]]:
        """Read a thread — prerequisite for classifying threaded emails."""
        if not email.thread_id:
            return (
                self._base_reward(action_cost=-0.01),
                "Email has no thread to read"
            )
        
        self.threads_read.add(email.thread_id)
        return self._base_reward(action_cost=-0.01), None
    
    def _handle_classify(self, action: Action, email: Email) -> Tuple[Reward, Optional[str]]:
        """Set category ONLY. Checks dependency (thread must be read first)."""
        if not action.category:
            return self._base_reward(action_cost=-0.02), "classify requires 'category' field"
        
        prog = self._get_or_create_progress(email.id)
        
        # Already classified?
        if prog.category is not None:
            return self._base_reward(action_cost=-0.02), "Email already classified"
        
        # Dependency check: threaded emails require read_thread first
        dep_penalty = 0.0
        if email.thread_id and email.thread_id not in self.threads_read:
            dep_penalty = -1.0
            self.dependency_violations += 1
        
        # New: Cross-email dependency check
        if email.depends_on:
            for dep_id in email.depends_on:
                dep_email = self._get_email(dep_id)
                dep_prog = self.progress.get(dep_id)
                
                # If dependency not processed or incorrectly classified
                if not dep_prog or not dep_prog.category or dep_prog.category != dep_email.true_category:
                    dep_penalty -= 0.5
                    self.dependency_violations += 1
        
        # Category accuracy
        CATEGORY_WEIGHTS = {
            EmailCategory.URGENT_ESCALATION: 3.0,
            EmailCategory.BILLING_ISSUE: 2.5,
            EmailCategory.TECHNICAL_SUPPORT: 2.0,
            EmailCategory.MEETING_REQUEST: 1.5,
            EmailCategory.SALES_INQUIRY: 1.0,
            EmailCategory.GENERAL_INFO: 0.8,
            EmailCategory.SPAM: 0.5,
            EmailCategory.INTERNAL: 0.8,
        }
        
        if action.category == email.true_category:
            cat_reward = CATEGORY_WEIGHTS[email.true_category]
        else:
            cat_reward = -1.0
        
        # Record
        prog.category = action.category
        prog.classified_step = self.current_step
        
        return Reward(
            total=cat_reward + dep_penalty + (-0.02),
            category_correct=cat_reward,
            priority_correct=0, routing_correct=0,
            sla_penalty=0, action_cost=-0.02,
            dependency_violation=dep_penalty, completion_bonus=0
        ), None
    
    def _handle_set_priority(self, action: Action, email: Email) -> Tuple[Reward, Optional[str]]:
        """Set priority ONLY. Must classify first."""
        if not action.priority:
            return self._base_reward(action_cost=-0.02), "set_priority requires 'priority' field"
        
        prog = self._get_or_create_progress(email.id)
        
        # Must classify before setting priority
        if prog.category is None:
            return (
                Reward(total=-1.0, category_correct=0, priority_correct=0,
                       routing_correct=0, sla_penalty=0, action_cost=-0.02,
                       dependency_violation=-0.98, completion_bonus=0),
                "Must classify email before setting priority"
            )
        
        # Already prioritized?
        if prog.priority is not None:
            return self._base_reward(action_cost=-0.02), "Priority already set"
        
        # Priority accuracy
        if action.priority == email.true_priority:
            pri_reward = 1.0
        else:
            pri_reward = -0.3
        
        prog.priority = action.priority
        prog.priority_step = self.current_step
        
        return Reward(
            total=pri_reward + (-0.02),
            category_correct=0, priority_correct=pri_reward,
            routing_correct=0, sla_penalty=0, action_cost=-0.02,
            dependency_violation=0, completion_bonus=0
        ), None
    
    def _handle_route(self, action: Action, email: Email) -> Tuple[Reward, Optional[str]]:
        """Route to a team. Terminal action. Must have category + priority first."""
        if not action.team:
            return self._base_reward(action_cost=-0.02), "route requires 'team' field"
        
        prog = self._get_or_create_progress(email.id)
        
        # Check prerequisites
        error = self._check_terminal_prereqs(prog, email.id)
        if error:
            return (
                Reward(total=-1.0, category_correct=0, priority_correct=0,
                       routing_correct=0, sla_penalty=0, action_cost=-0.02,
                       dependency_violation=-0.98, completion_bonus=0),
                error
            )
        
        # Routing accuracy
        if action.team == email.true_team:
            route_reward = 1.0
        else:
            route_reward = -0.5
        
        # Dependency check (again, to ensure terminal state respect dependencies)
        dep_penalty = 0.0
        if email.depends_on:
            for dep_id in email.depends_on:
                dep_prog = self.progress.get(dep_id)
                if not dep_prog or not dep_prog.is_done:
                    dep_penalty -= 0.5
                    self.dependency_violations += 1
        
        # SLA check
        sla_penalty = self._check_sla(email)
        
        # Completion bonus
        completion_bonus = self._completion_bonus(email)
        
        # Finalize
        prog.team = action.team
        prog.disposition = EmailDisposition.ROUTED
        prog.disposition_step = self.current_step
        
        total = route_reward + sla_penalty + (-0.02) + completion_bonus + dep_penalty
        return Reward(
            total=total, category_correct=0, priority_correct=0,
            routing_correct=route_reward, sla_penalty=sla_penalty,
            action_cost=-0.02, dependency_violation=dep_penalty,
            completion_bonus=completion_bonus
        ), None
    
    def _handle_archive(self, action: Action, email: Email) -> Tuple[Reward, Optional[str]]:
        """Archive email. Terminal action. Must have category + priority first."""
        prog = self._get_or_create_progress(email.id)
        
        error = self._check_terminal_prereqs(prog, email.id)
        if error:
            return (
                Reward(total=-1.0, category_correct=0, priority_correct=0,
                       routing_correct=0, sla_penalty=0, action_cost=-0.01,
                       dependency_violation=-0.99, completion_bonus=0),
                error
            )
        
        # Archiving is correct only for spam / general_info
        if email.true_category in [EmailCategory.SPAM, EmailCategory.GENERAL_INFO]:
            route_reward = 1.0
        else:
            route_reward = -1.0  # Shouldn't archive actionable emails
        
        # Dependency check
        dep_penalty = 0.0
        if email.depends_on:
            for dep_id in email.depends_on:
                dep_prog = self.progress.get(dep_id)
                if not dep_prog or not dep_prog.is_done:
                    dep_penalty -= 0.5
                    self.dependency_violations += 1
        
        sla_penalty = self._check_sla(email)
        completion_bonus = self._completion_bonus(email)
        
        prog.disposition = EmailDisposition.ARCHIVED
        prog.disposition_step = self.current_step
        
        total = route_reward + sla_penalty + (-0.01) + completion_bonus + dep_penalty
        return Reward(
            total=total, category_correct=0, priority_correct=0,
            routing_correct=route_reward, sla_penalty=sla_penalty,
            action_cost=-0.01, dependency_violation=dep_penalty,
            completion_bonus=completion_bonus
        ), None
    
    def _handle_escalate(self, action: Action, email: Email) -> Tuple[Reward, Optional[str]]:
        """Escalate email. Terminal action. Must have category + priority first."""
        prog = self._get_or_create_progress(email.id)
        
        error = self._check_terminal_prereqs(prog, email.id)
        if error:
            return (
                Reward(total=-1.0, category_correct=0, priority_correct=0,
                       routing_correct=0, sla_penalty=0, action_cost=-0.5,
                       dependency_violation=-0.5, completion_bonus=0),
                error
            )
        
        # Escalation is correct for urgent_escalation category
        if email.true_category == EmailCategory.URGENT_ESCALATION:
            route_reward = 2.0
        else:
            route_reward = -1.5  # Unnecessary escalation is costly
        
        # Dependency check
        dep_penalty = 0.0
        if email.depends_on:
            for dep_id in email.depends_on:
                dep_prog = self.progress.get(dep_id)
                if not dep_prog or not dep_prog.is_done:
                    dep_penalty -= 0.5
                    self.dependency_violations += 1
        
        sla_penalty = self._check_sla(email)
        completion_bonus = self._completion_bonus(email)
        
        prog.disposition = EmailDisposition.ESCALATED
        prog.disposition_step = self.current_step
        
        total = route_reward + sla_penalty + (-0.5) + completion_bonus + dep_penalty
        return Reward(
            total=total, category_correct=0, priority_correct=0,
            routing_correct=route_reward, sla_penalty=sla_penalty,
            action_cost=-0.5, dependency_violation=dep_penalty,
            completion_bonus=completion_bonus
        ), None
    
    # ─── Helper Methods ─────────────────────────────────────────────────
    
    def _check_terminal_prereqs(self, prog: EmailProgress, email_id: str) -> Optional[str]:
        """Terminal actions (route/archive/escalate) require classify + set_priority."""
        if prog.category is None:
            return f"Must classify email {email_id} before terminal action"
        if prog.priority is None:
            return f"Must set priority on email {email_id} before terminal action"
        if prog.disposition is not None:
            return f"Email {email_id} already has a terminal disposition"
        return None
    
    def _check_sla(self, email: Email) -> float:
        """Check SLA and return penalty."""
        if email.true_category in [EmailCategory.BILLING_ISSUE, EmailCategory.URGENT_ESCALATION]:
            steps_elapsed = self.current_step - email.arrival_step
            sla_threshold = 10 if email.true_category == EmailCategory.URGENT_ESCALATION else 15
            
            if steps_elapsed > sla_threshold:
                self.sla_violations += 1
                return -2.0
        return 0.0
    
    def _completion_bonus(self, email: Email) -> float:
        """Bonus for fully completing an email quickly."""
        steps_elapsed = self.current_step - email.arrival_step
        # More bonus for faster completion
        if steps_elapsed <= 5:
            return 0.5
        elif steps_elapsed <= 10:
            return 0.25
        return 0.0
    
    def _base_reward(self, action_cost: float = 0.0) -> Reward:
        """A zero-reward with only action cost."""
        return Reward(
            total=action_cost, category_correct=0, priority_correct=0,
            routing_correct=0, sla_penalty=0, action_cost=action_cost,
            dependency_violation=0, completion_bonus=0
        )
    
    def _calculate_decay(self) -> float:
        """Decay penalty for unprocessed urgent emails (applied globally)."""
        decay = 0.0
        for email in self.emails:
            prog = self.progress.get(email.id)
            is_done = prog.is_done if prog else False
            if not is_done and email.arrival_step <= self.current_step:
                if email.true_category == EmailCategory.URGENT_ESCALATION:
                    decay -= 0.15
                elif email.true_category == EmailCategory.BILLING_ISSUE:
                    decay -= 0.1
        return decay
    
    def _get_observation(self, error_msg: Optional[str] = None) -> Observation:
        """Build observation (hide ground truth). Includes in_progress emails."""
        inbox = []
        in_progress = []
        done_count = 0
        
        for email in self.emails:
            if email.arrival_step > self.current_step:
                continue  # Not yet arrived
            
            prog = self.progress.get(email.id)
            
            if prog and prog.is_done:
                done_count += 1
                continue
            
            email_dict = {
                "id": email.id,
                "subject": email.subject,
                "sender": email.sender,
                "body": email.body,
                "thread_id": email.thread_id,
                "thread_context": email.thread_context if (email.thread_id in self.threads_read) else None,
                "depends_on": email.depends_on,
                "arrival_step": email.arrival_step,
                "thread_read": (email.thread_id in self.threads_read) if email.thread_id else None,
            }
            
            if prog:
                # Has partial progress — show what's been done
                in_progress.append({
                    **email_dict,
                    "category_set": prog.category.value if prog.category else None,
                    "priority_set": prog.priority is not None,
                    "disposition": prog.disposition.value if prog.disposition else None,
                })
            else:
                inbox.append(email_dict)
        
        return Observation(
            inbox=inbox,
            in_progress=in_progress,
            processed_count=done_count,
            current_step=self.current_step,
            max_steps=self.max_steps,
            total_cost=abs(self.cumulative_reward) if self.cumulative_reward < 0 else 0,
            sla_violations=self.sla_violations,
            threads_read=list(self.threads_read),
            last_action_error=error_msg,
        )
    
    def _is_naturally_done(self) -> bool:
        """Check if episode completed naturally."""
        # All emails fully done
        if all(
            self.progress.get(e.id) and self.progress[e.id].is_done
            for e in self.emails
        ):
            return True
        # Too many violations (early termination)
        if self.sla_violations > 10:
            return True
        return False
    
    def _get_or_create_progress(self, email_id: str) -> EmailProgress:
        """Get or create a progress tracker for an email."""
        if email_id not in self.progress:
            self.progress[email_id] = EmailProgress()
        return self.progress[email_id]
    
    def _get_email(self, email_id: str) -> Optional[Email]:
        """Find email by ID"""
        for email in self.emails:
            if email.id == email_id:
                return email
        return None
    
    def _generate_emails(self, task_id: str) -> List[Email]:
        """Load synthetic emails for task"""
        from .email_loader import EmailLoader
        loader = EmailLoader()
        return loader.load(task_id)

    # ─── Backward compatibility for graders ──────────────────────────────
    
    @property
    def processed(self) -> Dict[str, Dict]:
        """Backward-compatible view: only fully done emails, in old format."""
        result = {}
        for email_id, prog in self.progress.items():
            if prog.is_done:
                result[email_id] = {
                    "category": prog.category,
                    "priority": prog.priority,
                    "team": prog.team,
                    "step": prog.disposition_step,
                }
        return result
