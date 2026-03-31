from sklearn.metrics import f1_score
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import EmailTriageEnv

def grade_episode(task_id: str, env: 'EmailTriageEnv') -> float:
    """
    Grade completed episode (returns 0.0 - 1.0)
    
    Uses env.progress for fine-grained scoring:
      - Only fully done emails (category + priority + disposition) count toward accuracy.
      - Partial progress is penalized via a coverage factor.
    
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
    """Easy: Category accuracy + coverage"""
    cat_f1 = _calculate_category_f1(env)
    coverage = _calculate_coverage(env)
    
    # 70% category accuracy, 30% coverage
    return max(0.0, min(1.0, float(cat_f1 * 0.7 + coverage * 0.3)))

def _grade_medium(env: 'EmailTriageEnv') -> float:
    """Medium: Category + Priority + SLA + Coverage"""
    cat_score = _calculate_category_f1(env) * 0.30
    pri_score = _calculate_priority_accuracy(env) * 0.25
    sla_score = max(0.0, 1.0 - (env.sla_violations / 5.0)) * 0.20
    coverage = _calculate_coverage(env) * 0.25
    
    return max(0.0, min(1.0, float(cat_score + pri_score + sla_score + coverage)))

def _grade_hard(env: 'EmailTriageEnv') -> float:
    """Hard: All components"""
    cat_score = _calculate_category_f1(env) * 0.20
    pri_score = _calculate_priority_accuracy(env) * 0.15
    route_score = _calculate_routing_accuracy(env) * 0.20
    sla_score = max(0.0, 1.0 - (env.sla_violations / 10.0)) * 0.15
    dep_score = max(0.0, 1.0 - (env.dependency_violations / 5.0)) * 0.10
    coverage = _calculate_coverage(env) * 0.20
    
    return max(0.0, min(1.0, float(cat_score + pri_score + route_score + sla_score + dep_score + coverage)))

# ─── Component Metrics ──────────────────────────────────────────────────

def _calculate_coverage(env: 'EmailTriageEnv') -> float:
    """Fraction of emails that are fully done (category + priority + disposition)."""
    if not env.emails:
        return 0.0
    done = sum(1 for e in env.emails 
               if env.progress.get(e.id) and env.progress[e.id].is_done)
    return done / len(env.emails)

def _calculate_category_f1(env: 'EmailTriageEnv') -> float:
    """Calculate category F1 score over done emails."""
    true_cats = []
    pred_cats = []
    
    for e in env.emails:
        true_val = e.true_category.value if hasattr(e.true_category, 'value') else e.true_category
        true_cats.append(true_val)
        
        prog = env.progress.get(e.id)
        if prog and prog.category is not None:
            pred_val = prog.category.value if hasattr(prog.category, 'value') else prog.category
            pred_cats.append(pred_val)
        else:
            pred_cats.append("unknown")
    
    return float(f1_score(true_cats, pred_cats, average='macro', zero_division=0.0))

def _calculate_priority_accuracy(env: 'EmailTriageEnv') -> float:
    """Calculate priority accuracy over emails that have priority set."""
    correct = 0
    total = 0
    for email in env.emails:
        prog = env.progress.get(email.id)
        if prog and prog.priority is not None:
            total += 1
            if prog.priority == email.true_priority:
                correct += 1
    return float(correct / total) if total > 0 else 0.0

def _calculate_routing_accuracy(env: 'EmailTriageEnv') -> float:
    """Calculate routing/disposition accuracy over done emails."""
    correct = 0
    total = 0
    for email in env.emails:
        prog = env.progress.get(email.id)
        if prog and prog.is_done:
            total += 1
            # Check disposition correctness
            if prog.team and prog.team == email.true_team:
                correct += 1
            elif prog.disposition and prog.disposition.value == "archived":
                # Archive is correct for spam/general_info
                from .models import EmailCategory
                if email.true_category in [EmailCategory.SPAM, EmailCategory.GENERAL_INFO]:
                    correct += 1
            elif prog.disposition and prog.disposition.value == "escalated":
                from .models import EmailCategory
                if email.true_category == EmailCategory.URGENT_ESCALATION:
                    correct += 1
    return float(correct / total) if total > 0 else 0.0
