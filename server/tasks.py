TASK_CONFIGS = {
    "easy": {
        "name": "Basic Email Classification",
        "num_emails": 15,
        # 3 steps per email (classify + set_priority + archive/route) => 45 min
        "max_steps": 50,
        "categories": ["spam", "billing_issue", "technical_support", "general_info"],
        "has_dependencies": False,
        "has_sla": False,
        "difficulty_mix": {"easy": 0.7, "medium": 0.3, "hard": 0.0}
    },
    "medium": {
        "name": "Priority-Aware Triage",
        "num_emails": 25,
        # 3-4 steps per email => 75-100
        "max_steps": 100,
        "categories": "all",
        "has_dependencies": False,
        "has_sla": True,
        "difficulty_mix": {"easy": 0.4, "medium": 0.5, "hard": 0.1}
    },
    "hard": {
        "name": "Full Inbox Management",
        "num_emails": 30,
        # 4 steps per email (read_thread + classify + set_priority + route/escalate) => 120
        "max_steps": 130,
        "categories": "all",
        "has_dependencies": True,
        "has_sla": True,
        "has_routing": True,
        "difficulty_mix": {"easy": 0.2, "medium": 0.5, "hard": 0.3}
    }
}
