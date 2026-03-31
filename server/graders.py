from sklearn.metrics import f1_score
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import EmailTriageEnv

def grade_episode(task_id: str, env: 'EmailTriageEnv') -> float:
    """
    Grade completed episode (returns 0.0 - 1.0)
    
    Args:
        task_id: Task identifier
        env: Environment instance with final state
        
    Returns:
        Score between 0.0 and 1.0
    """
    
    if task_id == "easy":
        return _grade_easy(env)
    elif task_id == "medium":
        return _grade_medium(env)
    elif task_id == "hard":
        return _grade_hard(env)
    else:
        return 0.0

def _grade_easy(env: 'EmailTriageEnv') -> float:
    """Easy: Category accuracy only"""
    true_cats = []
    pred_cats = []
    
    for email in env.emails:
        true_cats.append(email.true_category.value if hasattr(email.true_category, 'value') else email.true_category)
        pred = env.processed.get(email.id, {}).get("category")
        pred_value = pred.value if hasattr(pred, 'value') else pred
        pred_cats.append(pred_value if pred_value else "unknown")
    
    # Macro F1 score
    score = f1_score(true_cats, pred_cats, average='macro', zero_division=0.0)
    
    return max(0.0, min(1.0, float(score)))

def _grade_medium(env: 'EmailTriageEnv') -> float:
    """Medium: Category + Priority + SLA"""
    # Category (40%)
    cat_score = _calculate_category_f1(env) * 0.4
    
    # Priority (30%)
    pri_score = _calculate_priority_accuracy(env) * 0.3
    
    # SLA (30%)
    sla_score = max(0.0, 1.0 - (env.sla_violations / 5.0)) * 0.3
    
    return max(0.0, min(1.0, float(cat_score + pri_score + sla_score)))

def _grade_hard(env: 'EmailTriageEnv') -> float:
    """Hard: All components"""
    cat_score = _calculate_category_f1(env) * 0.25
    pri_score = _calculate_priority_accuracy(env) * 0.20
    route_score = _calculate_routing_accuracy(env) * 0.20
    sla_score = max(0.0, 1.0 - (env.sla_violations / 10.0)) * 0.20
    dep_score = max(0.0, 1.0 - (env.dependency_violations / 5.0)) * 0.15
    
    return max(0.0, min(1.0, float(cat_score + pri_score + route_score + sla_score + dep_score)))

def _calculate_category_f1(env: 'EmailTriageEnv') -> float:
    """Calculate category F1 score"""
    true_cats = [e.true_category.value if hasattr(e.true_category, 'value') else e.true_category for e in env.emails]
    pred_cats = []
    for e in env.emails:
        pred = env.processed.get(e.id, {}).get("category")
        pred_value = pred.value if hasattr(pred, 'value') else pred
        pred_cats.append(pred_value if pred_value else "unknown")
    
    return float(f1_score(true_cats, pred_cats, average='macro', zero_division=0.0))

def _calculate_priority_accuracy(env: 'EmailTriageEnv') -> float:
    """Calculate priority accuracy"""
    correct = 0
    total = 0
    for email in env.emails:
        pred = env.processed.get(email.id, {}).get("priority")
        if pred:
            total += 1
            if pred == email.true_priority:
                correct += 1
    return float(correct / total) if total > 0 else 0.0

def _calculate_routing_accuracy(env: 'EmailTriageEnv') -> float:
    """Calculate routing accuracy"""
    correct = 0
    total = 0
    for email in env.emails:
        pred = env.processed.get(email.id, {}).get("team")
        if pred:
            total += 1
            if pred == email.true_team:
                correct += 1
    return float(correct / total) if total > 0 else 0.0
