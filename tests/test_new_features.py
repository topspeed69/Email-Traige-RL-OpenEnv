import unittest
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.environment import EmailTriageEnv
from server.models import Action, EmailCategory

class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        self.env = EmailTriageEnv()
        self.env.reset("hard")
    
    def test_thread_context_visibility(self):
        """Verify that thread_context is hidden until read_thread is called."""
        obs = self.env.state()
        
        # Find a threaded email with context
        threaded_email = None
        for e in obs.inbox:
            if e.get("thread_id") == "thread_A":
                threaded_email = e
                break
        
        self.assertIsNotNone(threaded_email)
        self.assertIsNone(threaded_email.get("thread_context"))
        
        # Read the thread
        action = Action(action_type="read_thread", email_id=threaded_email["id"])
        self.env.step(action)
        
        # Check observation again
        obs = self.env.state()
        for e in obs.inbox + obs.in_progress:
            if e.get("thread_id") == "thread_A":
                self.assertIsNotNone(e.get("thread_context"))
                print(f"Revealed context: {e.get('thread_context')}")
                break

    def test_dependency_penalty(self):
        """Verify that missing/incorrect dependencies trigger penalties."""
        # Find an email with dependencies
        # In hard task, we can check email_m04 (added dependency on m11 if it exists in hard? 
        # No, m04 is in medium. Let's reset to medium.)
        self.env.reset("medium")
        obs = self.env.state()
        
        dep_email_id = "email_m04" # Depends on m11
        
        # Try to classify email_m04 without doing m11
        action = Action(
            action_type="classify", 
            email_id=dep_email_id, 
            category=EmailCategory.SALES_INQUIRY
        )
        obs, reward, _, _, _ = self.env.step(action)
        
        self.assertLess(reward.dependency_violation, 0)
        print(f"Dependency penalty for unfulfilled dep: {reward.dependency_violation}")

if __name__ == "__main__":
    unittest.main()
