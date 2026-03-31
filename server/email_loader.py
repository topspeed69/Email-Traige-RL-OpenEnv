from typing import List, Dict, Any
import json
import os
from .models import Email, EmailCategory

class EmailLoader:
    """Loads a static, deterministic dataset of synthetic emails for reproducibility."""
    
    def __init__(self):
        dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def load(self, task_id: str) -> List[Email]:
        emails_data = self.dataset.get(task_id, [])
        if not emails_data:
            # Fallback for when the full dataset isn't populated yet
            return []
            
        emails = []
        for e_data in emails_data:
            emails.append(Email(**e_data))
            
        # Keep them sorted by arrival step
        emails.sort(key=lambda x: x.arrival_step)
        
        return emails
