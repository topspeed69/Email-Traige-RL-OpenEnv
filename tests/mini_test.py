"""
Mini integration test for the multi-step email triage environment.
Tests the full pipeline directly (no HTTP server needed) with targeted assertions.

Run: .venv\Scripts\python -m pytest tests/mini_test.py -v
"""
from server.environment import EmailTriageEnv
from server.models import Action, EmailCategory
from server.graders import grade_episode


def _make(action_type, email_id, **kwargs):
    return Action(action_type=action_type, email_id=email_id, **kwargs)


class TestMultiStepPipeline:
    """Test the classify → set_priority → terminal action pipeline."""

    def setup_method(self):
        self.env = EmailTriageEnv()
        self.env.reset("easy")

    def test_full_pipeline_classify_priority_archive(self):
        """Spam email: classify → set_priority → archive (correct)."""
        eid = "email_e01"  # spam, low, support
        obs, r1, *_ = self.env.step(_make("classify", eid, category=EmailCategory.SPAM))
        assert r1.category_correct > 0, "Correct category should give positive reward"
        assert len(obs.in_progress) == 1, "Email should be in-progress"

        obs, r2, *_ = self.env.step(_make("set_priority", eid, priority="low"))
        assert r2.priority_correct > 0, "Correct priority should give positive reward"

        obs, r3, done, *_ = self.env.step(_make("archive", eid))
        assert r3.routing_correct > 0, "Archiving spam should be correct"
        assert r3.completion_bonus > 0, "Fast completion should earn bonus"
        assert obs.processed_count == 1
        assert eid not in [e["id"] for e in obs.inbox]
        assert eid not in [e["id"] for e in obs.in_progress]

    def test_wrong_category_penalty(self):
        """Misclassifying should give negative reward."""
        eid = "email_e01"  # true = spam
        _, r, *_ = self.env.step(_make("classify", eid, category=EmailCategory.BILLING_ISSUE))
        assert r.category_correct < 0

    def test_prerequisite_priority_before_classify(self):
        """set_priority before classify should fail."""
        eid = "email_e01"
        obs, r, *_ = self.env.step(_make("set_priority", eid, priority="high"))
        assert obs.last_action_error is not None
        assert "classify" in obs.last_action_error.lower()
        assert r.total < 0

    def test_prerequisite_route_before_priority(self):
        """route before set_priority should fail."""
        eid = "email_e01"
        self.env.step(_make("classify", eid, category=EmailCategory.SPAM))
        obs, r, *_ = self.env.step(_make("route", eid, team="support"))
        assert obs.last_action_error is not None
        assert "priority" in obs.last_action_error.lower()

    def test_double_classify_rejected(self):
        """Cannot classify same email twice."""
        eid = "email_e01"
        self.env.step(_make("classify", eid, category=EmailCategory.SPAM))
        obs, _, *_ = self.env.step(_make("classify", eid, category=EmailCategory.BILLING_ISSUE))
        assert obs.last_action_error is not None
        assert "already" in obs.last_action_error.lower()

    def test_double_terminal_rejected(self):
        """Cannot archive/route/escalate a done email."""
        eid = "email_e01"
        self.env.step(_make("classify", eid, category=EmailCategory.SPAM))
        self.env.step(_make("set_priority", eid, priority="low"))
        self.env.step(_make("archive", eid))
        obs, _, *_ = self.env.step(_make("archive", eid))
        assert obs.last_action_error is not None

    def test_invalid_email_id(self):
        """Bad email ID should return error reward."""
        obs, r, *_ = self.env.step(_make("classify", "nonexistent", category=EmailCategory.SPAM))
        assert r.total == -0.5
        assert obs.last_action_error is not None


class TestThreadDependency:
    """Test thread read_thread dependency for hard task."""

    def setup_method(self):
        self.env = EmailTriageEnv()
        self.env.reset("hard")

    def test_classify_without_read_thread_penalty(self):
        """Classifying a threaded email without read_thread should penalize."""
        eid = "email_h01"  # thread_A
        _, r, *_ = self.env.step(_make("classify", eid, category=EmailCategory.GENERAL_INFO))
        assert r.dependency_violation < 0
        assert self.env.dependency_violations == 1

    def test_read_thread_then_classify_no_penalty(self):
        """read_thread then classify should have no dependency penalty."""
        eid = "email_h01"  # thread_A
        self.env.step(_make("read_thread", eid))
        _, r, *_ = self.env.step(_make("classify", eid, category=EmailCategory.GENERAL_INFO))
        assert r.dependency_violation == 0

    def test_thread_read_status_in_observation(self):
        """Observation should expose thread_read status."""
        obs = self.env.state()
        threaded = [e for e in obs.inbox if e.get("thread_id")]
        assert len(threaded) > 0, "Hard task should have threaded emails"
        assert threaded[0].get("thread_read") is False

        # Read the thread
        self.env.step(_make("read_thread", threaded[0]["id"]))
        obs = self.env.state()
        tid = threaded[0]["thread_id"]
        assert tid in obs.threads_read

        # Emails in same thread should now show thread_read=True
        same_thread = [e for e in obs.inbox if e.get("thread_id") == tid]
        for e in same_thread:
            assert e["thread_read"] is True


class TestTerminalActions:
    """Test route / archive / escalate correctness."""

    def _process_to_terminal(self, eid, category, priority):
        self.env.step(_make("classify", eid, category=category))
        self.env.step(_make("set_priority", eid, priority=priority))

    def setup_method(self):
        self.env = EmailTriageEnv()
        self.env.reset("easy")

    def test_route_correct_team(self):
        eid = "email_e02"  # billing_issue, medium, finance
        self._process_to_terminal(eid, EmailCategory.BILLING_ISSUE, "medium")
        _, r, *_ = self.env.step(_make("route", eid, team="finance"))
        assert r.routing_correct > 0

    def test_route_wrong_team(self):
        eid = "email_e02"
        self._process_to_terminal(eid, EmailCategory.BILLING_ISSUE, "medium")
        _, r, *_ = self.env.step(_make("route", eid, team="engineering"))
        assert r.routing_correct < 0

    def test_archive_actionable_email_penalty(self):
        """Archiving an email that should be routed should penalize."""
        eid = "email_e02"  # billing_issue — should NOT be archived
        self._process_to_terminal(eid, EmailCategory.BILLING_ISSUE, "medium")
        _, r, *_ = self.env.step(_make("archive", eid))
        assert r.routing_correct < 0

    def test_escalate_correct(self):
        """Escalating an urgent_escalation email."""
        self.env.reset("medium")
        eid = "email_m01"  # urgent_escalation, high, engineering
        self._process_to_terminal(eid, EmailCategory.URGENT_ESCALATION, "high")
        _, r, *_ = self.env.step(_make("escalate", eid))
        assert r.routing_correct > 0

    def test_escalate_unnecessary_penalty(self):
        """Escalating a non-urgent email should penalize."""
        eid = "email_e01"  # spam
        self._process_to_terminal(eid, EmailCategory.SPAM, "low")
        _, r, *_ = self.env.step(_make("escalate", eid))
        assert r.routing_correct < 0


class TestGrading:
    """Test that graders work with the multi-step model."""

    def test_easy_perfect_score(self):
        """Process all easy emails correctly, expect high grade."""
        env = EmailTriageEnv()
        env.reset("easy")
        ground_truth = {e.id: e for e in env.emails}

        for email in env.emails:
            if email.thread_id:
                env.step(_make("read_thread", email.id))
            env.step(_make("classify", email.id, category=email.true_category))
            env.step(_make("set_priority", email.id, priority=email.true_priority))
            # Choose correct terminal action
            if email.true_category in [EmailCategory.SPAM, EmailCategory.GENERAL_INFO]:
                env.step(_make("archive", email.id))
            else:
                env.step(_make("route", email.id, team=email.true_team))

        score = grade_episode("easy", env)
        assert score > 0.85, f"Perfect play should score >0.85, got {score:.3f}"

    def test_medium_grade_nonzero(self):
        """Even naive play should produce a nonzero grade."""
        env = EmailTriageEnv()
        env.reset("medium")
        for email in env.emails:
            env.step(_make("classify", email.id, category=EmailCategory.GENERAL_INFO))
            env.step(_make("set_priority", email.id, priority="medium"))
            env.step(_make("archive", email.id))

        score = grade_episode("medium", env)
        assert 0.0 < score < 1.0, f"Naive score should be between 0 and 1, got {score:.3f}"


class TestObservation:
    """Test observation structure."""

    def test_inbox_vs_in_progress_separation(self):
        env = EmailTriageEnv()
        env.reset("easy")
        eid = env.emails[0].id

        obs = env.state()
        assert any(e["id"] == eid for e in obs.inbox)
        assert not any(e["id"] == eid for e in obs.in_progress)

        # Classify moves it to in_progress
        obs, *_ = env.step(_make("classify", eid, category=EmailCategory.SPAM))
        assert not any(e["id"] == eid for e in obs.inbox)
        assert any(e["id"] == eid for e in obs.in_progress)

        # Complete it
        env.step(_make("set_priority", eid, priority="low"))
        obs, *_ = env.step(_make("archive", eid))
        assert not any(e["id"] == eid for e in obs.inbox)
        assert not any(e["id"] == eid for e in obs.in_progress)
        assert obs.processed_count >= 1

    def test_episode_terminates_when_all_done(self):
        env = EmailTriageEnv()
        env.reset("easy")
        for email in env.emails:
            env.step(_make("classify", email.id, category=email.true_category))
            env.step(_make("set_priority", email.id, priority=email.true_priority))
            if email.true_category in [EmailCategory.SPAM, EmailCategory.GENERAL_INFO]:
                _, _, done, *_ = env.step(_make("archive", email.id))
            else:
                _, _, done, *_ = env.step(_make("route", email.id, team=email.true_team))

        assert done is True, "Episode should be done when all emails processed"
