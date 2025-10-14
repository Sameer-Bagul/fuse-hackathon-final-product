from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import time
import json
from collections import defaultdict, deque
import numpy as np

class LearningStrategy(Enum):
    """Different learning strategies that can be employed"""
    EXPLORATION_FOCUSED = "exploration_focused"
    EXPLOITATION_FOCUSED = "exploitation_focused"
    BALANCED = "balanced"
    CURRICULUM_DRIVEN = "curriculum_driven"
    REWARD_OPTIMIZED = "reward_optimized"
    ADAPTIVE = "adaptive"

@dataclass
class AdaptationRule:
    """Rule for adapting learning parameters based on conditions"""
    rule_id: str
    name: str
    description: str
    condition: Dict[str, Any]  # Conditions that trigger this rule
    actions: List[Dict[str, Any]]  # Actions to take when condition is met
    priority: int = 1  # Higher priority rules are checked first
    cooldown: float = 0.0  # Minimum time between applications (seconds)
    last_applied: float = 0.0

    def should_apply(self, metrics: Dict[str, Any]) -> bool:
        """Check if this rule's conditions are met"""
        try:
            condition_type = self.condition.get('type', 'threshold')

            if condition_type == 'threshold':
                metric_name = self.condition['metric']
                operator = self.condition['operator']
                threshold = self.condition['threshold']

                if metric_name not in metrics:
                    return False

                value = metrics[metric_name]

                if operator == 'gt':
                    return value > threshold
                elif operator == 'lt':
                    return value < threshold
                elif operator == 'gte':
                    return value >= threshold
                elif operator == 'lte':
                    return value <= threshold
                elif operator == 'eq':
                    return value == threshold

            elif condition_type == 'range':
                metric_name = self.condition['metric']
                min_val = self.condition.get('min')
                max_val = self.condition.get('max')

                if metric_name not in metrics:
                    return False

                value = metrics[metric_name]
                return (min_val is None or value >= min_val) and (max_val is None or value <= max_val)

            elif condition_type == 'trend':
                # Check if metric is trending in a certain direction
                metric_name = self.condition['metric']
                direction = self.condition['direction']  # 'increasing', 'decreasing'
                window = self.condition.get('window', 5)

                if metric_name not in metrics or 'history' not in metrics:
                    return False

                history = metrics['history'].get(metric_name, [])
                if len(history) < window:
                    return False

                recent = history[-window:]
                if direction == 'increasing':
                    return all(recent[i] <= recent[i+1] for i in range(len(recent)-1))
                elif direction == 'decreasing':
                    return all(recent[i] >= recent[i+1] for i in range(len(recent)-1))

            return False

        except Exception as e:
            print(f"Error evaluating rule {self.rule_id}: {e}")
            return False

    def can_apply(self) -> bool:
        """Check if rule can be applied (cooldown check)"""
        if self.cooldown <= 0:
            return True
        return time.time() - self.last_applied >= self.cooldown

    def apply(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the rule's actions to current parameters"""
        new_params = current_params.copy()

        for action in self.actions:
            action_type = action.get('type')

            if action_type == 'set_parameter':
                param_name = action['parameter']
                value = action['value']
                new_params[param_name] = value

            elif action_type == 'adjust_parameter':
                param_name = action['parameter']
                adjustment = action['adjustment']
                operation = action.get('operation', 'add')

                if param_name in new_params:
                    current_value = new_params[param_name]
                    if operation == 'add':
                        new_params[param_name] = current_value + adjustment
                    elif operation == 'multiply':
                        new_params[param_name] = current_value * adjustment
                    elif operation == 'set':
                        new_params[param_name] = adjustment

            elif action_type == 'clamp_parameter':
                param_name = action['parameter']
                min_val = action.get('min')
                max_val = action.get('max')

                if param_name in new_params:
                    value = new_params[param_name]
                    if min_val is not None:
                        value = max(value, min_val)
                    if max_val is not None:
                        value = min(value, max_val)
                    new_params[param_name] = value

        self.last_applied = time.time()
        return new_params

@dataclass
class MetaMetrics:
    """Comprehensive metrics for meta-learning performance tracking"""
    strategy_performance: Dict[str, Dict[str, Any]]  # strategy -> performance data
    parameter_history: List[Dict[str, Any]]  # History of parameter changes
    adaptation_events: List[Dict[str, Any]]  # History of adaptations made
    performance_trends: Dict[str, List[float]]  # metric_name -> list of values over time
    context_awareness: Dict[str, Any]  # Current learning context
    last_updated: float

    def __init__(self):
        self.strategy_performance = {}
        self.parameter_history = []
        self.adaptation_events = []
        self.performance_trends = defaultdict(list)
        self.context_awareness = {}
        self.last_updated = time.time()

    def update_strategy_performance(self, strategy: LearningStrategy, metrics: Dict[str, Any]):
        """Update performance metrics for a specific strategy"""
        strategy_key = strategy.value
        if strategy_key not in self.strategy_performance:
            self.strategy_performance[strategy_key] = {
                'total_episodes': 0,
                'total_reward': 0.0,
                'avg_reward': 0.0,
                'success_rate': 0.0,
                'last_used': 0.0,
                'usage_count': 0
            }

        perf = self.strategy_performance[strategy_key]
        perf['total_episodes'] += 1
        perf['total_reward'] += metrics.get('reward', 0.0)
        perf['avg_reward'] = perf['total_reward'] / perf['total_episodes']
        perf['success_rate'] = metrics.get('success_rate', 0.0)
        perf['last_used'] = time.time()
        perf['usage_count'] += 1

        self.last_updated = time.time()

    def record_parameter_change(self, old_params: Dict[str, Any], new_params: Dict[str, Any],
                               reason: str, rule_id: Optional[str] = None):
        """Record a parameter change event"""
        change_event = {
            'timestamp': time.time(),
            'old_params': old_params.copy(),
            'new_params': new_params.copy(),
            'changes': {},
            'reason': reason,
            'rule_id': rule_id
        }

        # Calculate what changed
        for key in set(old_params.keys()) | set(new_params.keys()):
            old_val = old_params.get(key)
            new_val = new_params.get(key)
            if old_val != new_val:
                change_event['changes'][key] = {
                    'from': old_val,
                    'to': new_val
                }

        self.parameter_history.append(change_event)
        self.last_updated = time.time()

    def record_adaptation(self, rule_id: str, strategy: LearningStrategy,
                         old_params: Dict[str, Any], new_params: Dict[str, Any],
                         trigger_metrics: Dict[str, Any]):
        """Record an adaptation event"""
        adaptation_event = {
            'timestamp': time.time(),
            'rule_id': rule_id,
            'strategy': strategy.value,
            'old_params': old_params.copy(),
            'new_params': new_params.copy(),
            'trigger_metrics': trigger_metrics.copy(),
            'performance_impact': {}  # To be filled later
        }

        self.adaptation_events.append(adaptation_event)
        self.last_updated = time.time()

    def update_performance_trend(self, metric_name: str, value: float):
        """Update a performance trend metric"""
        self.performance_trends[metric_name].append(value)

        # Keep only last 100 values to prevent memory issues
        if len(self.performance_trends[metric_name]) > 100:
            self.performance_trends[metric_name] = self.performance_trends[metric_name][-100:]

        self.last_updated = time.time()

    def update_context(self, context: Dict[str, Any]):
        """Update the current learning context"""
        self.context_awareness.update(context)
        self.context_awareness['last_updated'] = time.time()
        self.last_updated = time.time()

    def get_best_strategy(self) -> Optional[LearningStrategy]:
        """Get the best performing strategy based on historical data"""
        if not self.strategy_performance:
            return None

        best_strategy = None
        best_score = -float('inf')

        for strategy_key, perf in self.strategy_performance.items():
            # Score based on average reward and usage count (prefer tried strategies)
            score = perf['avg_reward'] * (1 + np.log(perf['usage_count'] + 1) * 0.1)
            if score > best_score:
                best_score = score
                best_strategy = LearningStrategy(strategy_key)

        return best_strategy

    def get_recent_trends(self, metric_name: str, window: int = 10) -> List[float]:
        """Get recent trend data for a metric"""
        return self.performance_trends[metric_name][-window:] if metric_name in self.performance_trends else []

    def get_adaptation_effectiveness(self, rule_id: str, window: int = 5) -> float:
        """Calculate how effective a particular adaptation rule has been"""
        rule_adaptations = [event for event in self.adaptation_events
                           if event['rule_id'] == rule_id]

        if not rule_adaptations:
            return 0.0

        # Look at performance impact after adaptations
        total_impact = 0.0
        count = 0

        for adaptation in rule_adaptations[-window:]:  # Last N adaptations
            impact = adaptation.get('performance_impact', {}).get('reward_change', 0.0)
            total_impact += impact
            count += 1

        return total_impact / count if count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'strategy_performance': self.strategy_performance,
            'parameter_history': self.parameter_history,
            'adaptation_events': self.adaptation_events,
            'performance_trends': dict(self.performance_trends),
            'context_awareness': self.context_awareness,
            'last_updated': self.last_updated
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaMetrics':
        """Create from dictionary"""
        metrics = cls()
        metrics.strategy_performance = data.get('strategy_performance', {})
        metrics.parameter_history = data.get('parameter_history', [])
        metrics.adaptation_events = data.get('adaptation_events', [])
        metrics.performance_trends = defaultdict(list, data.get('performance_trends', {}))
        metrics.context_awareness = data.get('context_awareness', {})
        metrics.last_updated = data.get('last_updated', time.time())
        return metrics

class MetaLearner:
    """Meta-learning component that adapts learning strategies and parameters"""

    def __init__(self):
        self.current_strategy = LearningStrategy.BALANCED
        self.current_params = {
            'alpha': 0.1,  # Learning rate
            'epsilon': 0.1,  # Exploration rate
            'gamma': 0.99,  # Discount factor
            'strategy_weights': {
                'exploration': 0.5,
                'exploitation': 0.5
            }
        }

        self.metrics = MetaMetrics()
        self.adaptation_rules = self._create_default_rules()
        self.strategy_history = deque(maxlen=50)  # Recent strategy usage

    def _create_default_rules(self) -> List[AdaptationRule]:
        """Create default adaptation rules"""
        rules = []

        # Rule 1: High exploration when performance is poor
        rules.append(AdaptationRule(
            rule_id="high_exploration_on_low_performance",
            name="Increase Exploration on Low Performance",
            description="Increase epsilon when average reward is low",
            condition={
                'type': 'threshold',
                'metric': 'avg_reward',
                'operator': 'lt',
                'threshold': 0.3
            },
            actions=[
                {'type': 'adjust_parameter', 'parameter': 'epsilon', 'adjustment': 0.1, 'operation': 'add'},
                {'type': 'clamp_parameter', 'parameter': 'epsilon', 'min': 0.0, 'max': 0.8}
            ],
            priority=3,
            cooldown=30.0  # 30 seconds between applications
        ))

        # Rule 2: Reduce exploration when performance is good
        rules.append(AdaptationRule(
            rule_id="reduce_exploration_on_good_performance",
            name="Reduce Exploration on Good Performance",
            description="Decrease epsilon when average reward is high",
            condition={
                'type': 'threshold',
                'metric': 'avg_reward',
                'operator': 'gt',
                'threshold': 0.7
            },
            actions=[
                {'type': 'adjust_parameter', 'parameter': 'epsilon', 'adjustment': 0.05, 'operation': 'multiply'},
                {'type': 'clamp_parameter', 'parameter': 'epsilon', 'min': 0.01, 'max': 0.5}
            ],
            priority=3,
            cooldown=60.0  # 1 minute between applications
        ))

        # Rule 3: Increase learning rate when stuck
        rules.append(AdaptationRule(
            rule_id="increase_learning_rate_when_stuck",
            name="Increase Learning Rate When Stuck",
            description="Increase alpha when reward trend is flat",
            condition={
                'type': 'trend',
                'metric': 'avg_reward',
                'direction': 'decreasing',
                'window': 5
            },
            actions=[
                {'type': 'adjust_parameter', 'parameter': 'alpha', 'adjustment': 1.2, 'operation': 'multiply'},
                {'type': 'clamp_parameter', 'parameter': 'alpha', 'min': 0.01, 'max': 0.5}
            ],
            priority=2,
            cooldown=120.0  # 2 minutes between applications
        ))

        # Rule 4: Switch to curriculum-driven when curriculum progress is available
        rules.append(AdaptationRule(
            rule_id="switch_to_curriculum_driven",
            name="Switch to Curriculum-Driven Learning",
            description="Switch strategy when curriculum data is available",
            condition={
                'type': 'threshold',
                'metric': 'curriculum_completion',
                'operator': 'gte',
                'threshold': 0.1
            },
            actions=[
                {'type': 'set_parameter', 'parameter': 'strategy', 'value': LearningStrategy.CURRICULUM_DRIVEN}
            ],
            priority=1,
            cooldown=300.0  # 5 minutes between applications
        ))

        return rules

    def adapt_parameters(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt learning parameters based on current performance metrics"""
        # Update metrics
        self.metrics.update_context(current_metrics)

        # Update performance trends
        for metric_name, value in current_metrics.items():
            if isinstance(value, (int, float)):
                self.metrics.update_performance_trend(metric_name, value)

        # Check and apply adaptation rules
        applicable_rules = []
        for rule in sorted(self.adaptation_rules, key=lambda r: r.priority, reverse=True):
            if rule.should_apply(current_metrics) and rule.can_apply():
                applicable_rules.append(rule)

        # Apply highest priority rule
        if applicable_rules:
            rule = applicable_rules[0]
            old_params = self.current_params.copy()
            new_params = rule.apply(self.current_params)

            # Record the adaptation
            self.metrics.record_adaptation(
                rule.rule_id, self.current_strategy, old_params, new_params, current_metrics
            )

            # Update strategy if changed
            if 'strategy' in new_params and new_params['strategy'] != self.current_strategy:
                self.current_strategy = new_params['strategy']
                self.strategy_history.append(self.current_strategy)

            self.current_params = new_params

            # Record parameter change
            self.metrics.record_parameter_change(
                old_params, new_params, f"Applied rule: {rule.name}", rule.rule_id
            )

        return self.current_params.copy()

    def select_optimal_strategy(self, context: Dict[str, Any]) -> LearningStrategy:
        """Select the optimal learning strategy based on context and performance history"""
        # Consider context factors
        curriculum_available = context.get('curriculum_completion', 0) > 0
        performance_trend = self.metrics.get_recent_trends('avg_reward', 10)
        current_difficulty = context.get('difficulty', 'medium')

        # Strategy selection logic
        if curriculum_available and len(performance_trend) > 5:
            # Use curriculum-driven if curriculum is available and we have performance data
            recent_avg = sum(performance_trend[-5:]) / 5
            if recent_avg > 0.6:
                return LearningStrategy.CURRICULUM_DRIVEN

        if performance_trend:
            recent_performance = performance_trend[-1]
            if recent_performance < 0.4:
                return LearningStrategy.EXPLORATION_FOCUSED
            elif recent_performance > 0.8:
                return LearningStrategy.EXPLOITATION_FOCUSED

        # Default to adaptive strategy
        return LearningStrategy.ADAPTIVE

    def get_adaptation_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for parameter adaptations"""
        recommendations = []

        # Analyze recent performance
        reward_trend = self.metrics.get_recent_trends('avg_reward', 10)
        if len(reward_trend) >= 5:
            recent_avg = sum(reward_trend[-5:]) / 5
            overall_trend = reward_trend[-1] - reward_trend[0]

            if recent_avg < 0.5:
                recommendations.append({
                    'type': 'parameter_adjustment',
                    'parameter': 'epsilon',
                    'action': 'increase',
                    'reason': 'Low recent performance suggests need for more exploration',
                    'expected_impact': 'medium'
                })

            if overall_trend < 0:
                recommendations.append({
                    'type': 'strategy_change',
                    'new_strategy': 'exploration_focused',
                    'reason': 'Declining performance trend indicates current strategy ineffective',
                    'expected_impact': 'high'
                })

        return recommendations

    def get_meta_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning status"""
        return {
            'current_strategy': self.current_strategy.value,
            'current_params': self.current_params.copy(),
            'strategy_history': [s.value for s in self.strategy_history],
            'active_rules': [
                {
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'last_applied': rule.last_applied,
                    'can_apply': rule.can_apply()
                }
                for rule in self.adaptation_rules
            ],
            'performance_summary': {
                'best_strategy': self.metrics.get_best_strategy().value if self.metrics.get_best_strategy() else None,
                'total_adaptations': len(self.metrics.adaptation_events),
                'recent_trends': {
                    metric: self.metrics.get_recent_trends(metric, 5)
                    for metric in ['avg_reward', 'success_rate']
                }
            },
            'recommendations': self.get_adaptation_recommendations()
        }