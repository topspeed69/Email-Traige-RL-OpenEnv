TASK_CONFIGS = {
    "easy": {
        "name": "Basic Email Classification",
        "num_emails": 15,
        "max_steps": 20,
        "categories": ["spam", "billing_issue", "technical_support", "general_info"],
        "has_dependencies": False,
        "has_sla": False,
        "difficulty_mix": {"easy": 0.7, "medium": 0.3, "hard": 0.0}
    },
    "medium": {
        "name": "Priority-Aware Triage",
        "num_emails": 25,
        "max_steps": 40,
        "categories": "all",
        "has_dependencies": False,
        "has_sla": True,
        "difficulty_mix": {"easy": 0.4, "medium": 0.5, "hard": 0.1}
    },
    "hard": {
        "name": "Full Inbox Management",
        "num_emails": 30,
        "max_steps": 50,
        "categories": "all",
        "has_dependencies": True,
        "has_sla": True,
        "has_routing": True,
        "difficulty_mix": {"easy": 0.2, "medium": 0.5, "hard": 0.3}
    }
}
