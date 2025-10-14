"""
Visualization Service - Educational Implementation

This module demonstrates how to create data visualizations for machine learning systems.
Think of this as the "View" layer in MVC architecture - it takes raw data and transforms
it into formats that humans can easily understand and analyze.

Why do we need visualization?
- Raw numbers are hard to interpret quickly
- Visual patterns reveal insights that numbers hide
- Charts help communicate complex ideas to stakeholders
- Visual feedback accelerates the learning process

Design Patterns Used:
- Service Layer Pattern: Encapsulates visualization logic
- Builder Pattern: Constructs complex chart data structures
- Strategy Pattern: Different visualization strategies for different data types

Best Practices Demonstrated:
- Separation of concerns (data processing vs. presentation)
- Error handling for edge cases
- Flexible data formats (matplotlib for static, Chart.js for web)
- Configurable parameters for different use cases
"""

import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any, Optional, Union
import logging

# Configure logging to help students understand what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationService:
    """
    A comprehensive visualization service for LLM learning analytics.

    This class teaches you how to:
    1. Transform raw ML data into meaningful visualizations
    2. Handle different chart types and data formats
    3. Prepare data for both static plots and interactive web charts
    4. Implement proper error handling and logging

    Real-world applications:
    - Monitoring ML model performance over time
    - Analyzing user interaction patterns
    - Creating dashboards for stakeholders
    - Debugging learning algorithms
    """

    def __init__(self):
        """
        Initialize the visualization service.

        Why do we need an __init__ method?
        - Set up default configurations
        - Initialize any required resources
        - Configure logging or external connections
        """
        self.default_figsize = (12, 8)
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        logger.info("VisualizationService initialized with default settings")

    def plot_learning_curve(self, rewards: List[float], title: str = "Training Rewards") -> None:
        """
        Create a comprehensive learning curve visualization.

        This method demonstrates how to create multi-panel plots that show
        different aspects of the learning process simultaneously.

        Args:
            rewards: List of reward values from training episodes
            title: Title for the entire figure

        Why multiple subplots?
        - Learning is multi-dimensional - we want to see different metrics
        - Comparing before/after performance shows improvement
        - Smoothed curves help identify trends vs. noise
        - Difficulty progression shows curriculum learning

        Educational Notes:
        - Always consider what story your data is telling
        - Use appropriate chart types for different data types
        - Color coding helps distinguish different metrics
        - Titles and labels make charts self-explanatory
        """
        if not rewards:
            logger.warning("No reward data provided for plotting")
            return

        n = len(rewards)
        logger.info(f"Creating learning curve plot with {n} data points")

        # Create a 2x2 grid of subplots - this is a common pattern in data visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.default_figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Subplot 1: Raw learning curve
        # Why show raw data? To see the actual learning trajectory
        ax1.plot(rewards, 'b-', linewidth=2, label='Reward per Episode')
        ax1.set_title("Learning Curve (Raw Data)", fontsize=12)
        ax1.set_xlabel("Training Episode")
        ax1.set_ylabel("Reward Value")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Educational tip: Raw data shows volatility, which is normal in RL

        # Subplot 2: Before/After comparison
        # Why compare halves? To show learning progress over time
        half = n // 2
        if half > 0:
            avg_before = sum(rewards[:half]) / half
            avg_after = sum(rewards[half:]) / (n - half)

            ax2.bar(['First Half', 'Second Half'], [avg_before, avg_after],
                   color=['lightcoral', 'lightgreen'], alpha=0.7)
            ax2.set_title("Performance Improvement", fontsize=12)
            ax2.set_ylabel("Average Reward")

            # Add improvement percentage
            if avg_before > 0:
                improvement = ((avg_after - avg_before) / avg_before) * 100
                ax2.text(0.5, max(avg_before, avg_after) * 0.9,
                        f"Improvement: {improvement:.1f}%",
                        ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

        # Educational tip: Comparing averages helps quantify learning

        # Subplot 3: Difficulty progression (simulated curriculum learning)
        # Why show difficulty? Many ML systems use curriculum learning
        difficulty = [i / max(1, n - 1) for i in range(n)]  # Linear increase
        ax3.plot(difficulty, 'r--', linewidth=2, label='Difficulty Level')
        ax3.fill_between(range(n), 0, difficulty, alpha=0.3, color='red')
        ax3.set_title("Curriculum Difficulty Progression", fontsize=12)
        ax3.set_xlabel("Training Episode")
        ax3.set_ylabel("Difficulty (0-1)")
        ax3.legend()

        # Educational tip: Curriculum learning gradually increases task complexity

        # Subplot 4: Smoothed learning curve
        # Why smooth? To see trends by removing noise
        window_size = max(1, n // 20)  # Adaptive window size
        smoothed = self._moving_average(rewards, window_size)

        ax4.plot(smoothed, 'g-', linewidth=3, label=f'Smoothed (window={window_size})')
        ax4.plot(rewards, 'gray', alpha=0.5, linewidth=1, label='Raw data')
        ax4.set_title("Smoothed Learning Trend", fontsize=12)
        ax4.set_xlabel("Training Episode")
        ax4.set_ylabel("Smoothed Reward")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Educational tip: Smoothing helps identify if learning is actually happening

        plt.tight_layout()
        plt.show()
        logger.info("Learning curve visualization completed")

    def plot_learning_progress(self, metrics: Dict[str, Any], title: str = "LLM Learning Progress") -> None:
        """
        Create a progress visualization showing multiple learning metrics.

        This demonstrates how to combine different types of data in one visualization.

        Args:
            metrics: Dictionary containing various learning metrics
            title: Title for the visualization

        Educational Notes:
        - Different metrics need different chart types (line, bar, etc.)
        - Color coordination helps relate different subplots
        - Consistent scaling makes comparisons easier
        """
        logger.info(f"Creating learning progress plot with metrics: {list(metrics.keys())}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Subplot 1: Success rate over time
        if 'success_rates' in metrics and metrics['success_rates']:
            success_rates = metrics['success_rates']
            axes[0].plot(success_rates, 'b-o', linewidth=2, markersize=4, alpha=0.8)
            axes[0].set_title("Success Rate Progression", fontsize=12)
            axes[0].set_xlabel("Interaction Number")
            axes[0].set_ylabel("Success Rate (0-1)")
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 1.1)  # Success rate should be 0-1

            # Add trend line
            if len(success_rates) > 1:
                from numpy import polyfit
                try:
                    z = polyfit(range(len(success_rates)), success_rates, 1)
                    p = [z[0]*i + z[1] for i in range(len(success_rates))]
                    axes[0].plot(p, 'r--', alpha=0.7, label='Trend')
                    axes[0].legend()
                except:
                    pass  # Polyfit might fail with insufficient data

        # Subplot 2: Pattern frequency analysis
        if 'patterns' in metrics and metrics['patterns']:
            patterns = metrics['patterns']
            # Get top patterns for readability
            top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            if top_patterns:
                words, freqs = zip(*top_patterns)
                bars = axes[1].bar(range(len(words)), freqs, color='skyblue', alpha=0.8)
                axes[1].set_title("Most Frequent Prompt Patterns", fontsize=12)
                axes[1].set_xlabel("Pattern")
                axes[1].set_ylabel("Frequency")
                axes[1].set_xticks(range(len(words)))
                axes[1].set_xticklabels(words, rotation=45, ha='right')

                # Add value labels on bars
                for bar, freq in zip(bars, freqs):
                    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{int(freq)}', ha='center', va='bottom', fontsize=9)

        # Subplot 3: Task generation metrics
        if 'task_metrics' in metrics and metrics['task_metrics']:
            task_data = metrics['task_metrics']
            axes[2].plot(task_data, 'g-s', linewidth=2, markersize=6, alpha=0.8)
            axes[2].set_title("Task Generation Performance", fontsize=12)
            axes[2].set_xlabel("Task Number")
            axes[2].set_ylabel("Performance Metric")
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        logger.info("Learning progress visualization completed")

    def get_chartjs_data(self, rewards: List[float], metrics: Optional[Dict[str, Any]] = None, learning_loop_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare data in Chart.js format for web dashboards.

        This method shows how to transform Python data structures into
        JSON format suitable for JavaScript charting libraries.

        Args:
            rewards: List of reward values
            metrics: Optional additional metrics
            learning_loop_data: Optional learning loop progress data

        Returns:
            Dictionary containing Chart.js compatible data

        Educational Notes:
        - Web dashboards need JSON data, not matplotlib plots
        - Chart.js expects specific data structure (labels, datasets)
        - Colors and styling are specified in the data structure
        - This enables interactive web visualizations

        Best Practices:
        - Validate data before creating charts
        - Use consistent color schemes
        - Include proper labels and legends
        - Handle missing data gracefully
        """
        logger.info("Preparing Chart.js data for web visualization")

        # Use learning loop rewards if available, otherwise use provided rewards
        if learning_loop_data and 'recent_rewards' in learning_loop_data and learning_loop_data['recent_rewards']:
            display_rewards = learning_loop_data['recent_rewards']
            reward_label = 'Autonomous Learning Rewards'
        elif rewards:
            display_rewards = rewards
            reward_label = 'Training Reward'
        else:
            logger.info("Empty state: No reward data provided, returning empty array")
            return []

        # Base learning curve data
        chart_data = {
            'learning_curve': {
                'labels': [f'Iteration {i+1}' for i in range(len(display_rewards))],
                'datasets': [{
                    'label': reward_label,
                    'data': display_rewards,
                    'borderColor': 'rgb(75, 192, 192)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                    'tension': 0.4,
                    'fill': True
                }]
            }
        }

        # Add average reward trend if learning loop data available
        if learning_loop_data and 'average_rewards' in learning_loop_data and learning_loop_data['average_rewards']:
            avg_rewards = learning_loop_data['average_rewards']
            chart_data['learning_curve']['datasets'].append({
                'label': 'Average Reward Trend',
                'data': avg_rewards,
                'borderColor': 'rgb(255, 159, 64)',
                'backgroundColor': 'rgba(255, 159, 64, 0.2)',
                'tension': 0.4,
                'fill': False,
                'borderDash': [5, 5]
            })

        # Add success rate data if available
        if learning_loop_data and 'success_rates' in learning_loop_data and learning_loop_data['success_rates']:
            success_rates = learning_loop_data['success_rates']
            chart_data['success_rate'] = {
                'labels': [f'Iteration {i+1}' for i in range(len(success_rates))],
                'datasets': [{
                    'label': 'Success Rate',
                    'data': success_rates,
                    'borderColor': 'rgb(255, 99, 132)',
                    'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                    'tension': 0.4,
                    'fill': False
                }]
            }
        elif metrics and 'success_rates' in metrics and metrics['success_rates']:
            success_rates = metrics['success_rates']
            chart_data['success_rate'] = {
                'labels': [f'Interaction {i+1}' for i in range(len(success_rates))],
                'datasets': [{
                    'label': 'Success Rate',
                    'data': success_rates,
                    'borderColor': 'rgb(255, 99, 132)',
                    'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                    'tension': 0.4,
                    'fill': False
                }]
            }

        # Add pattern frequency data if available
        if metrics and 'patterns' in metrics and metrics['patterns']:
            patterns = metrics['patterns']
            top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            if top_patterns:
                words, freqs = zip(*top_patterns)
                chart_data['patterns'] = {
                    'labels': list(words),
                    'datasets': [{
                        'label': 'Pattern Frequency',
                        'data': list(freqs),
                        'backgroundColor': 'rgba(54, 162, 235, 0.8)',
                        'borderColor': 'rgba(54, 162, 235, 1)',
                        'borderWidth': 1
                    }]
                }

        # Add learning loop specific metrics
        if learning_loop_data:
            # Iterations over time
            if 'iterations' in learning_loop_data:
                chart_data['iterations'] = {
                    'labels': ['Current'],
                    'datasets': [{
                        'label': 'Total Iterations',
                        'data': [learning_loop_data['iterations']],
                        'backgroundColor': 'rgba(153, 102, 255, 0.8)',
                        'borderColor': 'rgba(153, 102, 255, 1)',
                        'borderWidth': 1
                    }]
                }

        # Transform to array format for Recharts compatibility
        # Return array of data points with keys that frontend components expect
        array_data = []
        for i, reward in enumerate(display_rewards):
            data_point = {
                'iteration': i + 1,
                'reward': reward,
                'predicted': reward,  # Use current reward as predicted for demo
                'confidence': 0.95,  # Default confidence
                'successRate': min(100, reward * 100 + 50),  # Transform to percentage
                'interactions': (i + 1) * 10,  # Mock interaction count
                'time': f'{i:02d}:00',  # Time format
                'day': f'Iteration {i + 1}'  # Day format
            }
            array_data.append(data_point)

        logger.info(f"Chart data prepared as array with {len(array_data)} data points")
        return array_data

    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """
        Calculate moving average to smooth noisy data.

        This is a fundamental signal processing technique that helps
        identify trends by reducing random fluctuations.

        Args:
            data: Input data series
            window_size: Size of the moving window

        Returns:
            Smoothed data series

        Educational Notes:
        - Moving averages reduce noise while preserving trends
        - Window size affects smoothness vs. responsiveness
        - Edge effects occur at the beginning and end
        - This is used in technical analysis, signal processing, etc.
        """
        if not data or window_size <= 0:
            return data

        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            window = data[start:end]
            smoothed.append(sum(window) / len(window))

        return smoothed

    def create_comprehensive_report(self, rewards: List[float], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive learning report with multiple visualizations.

        This method demonstrates how to combine multiple analysis techniques
        into a single, coherent report.

        Args:
            rewards: Training reward data
            metrics: Learning metrics

        Returns:
            Dictionary containing various analysis results

        Educational Notes:
        - Comprehensive analysis requires multiple perspectives
        - Combining different metrics gives deeper insights
        - Reports should be self-contained and interpretable
        - This is how real ML engineers analyze model performance
        """
        logger.info("Generating comprehensive learning report")

        report = {
            'summary': {
                'total_episodes': len(rewards),
                'average_reward': sum(rewards) / len(rewards) if rewards else 0,
                'max_reward': max(rewards) if rewards else 0,
                'min_reward': min(rewards) if rewards else 0,
                'total_interactions': metrics.get('total_interactions', 0),
                'success_rate': metrics.get('success_rate', 0)
            },
            'charts': {
                'learning_curve': self.get_chartjs_data(rewards, metrics)
            },
            'insights': self._generate_insights(rewards, metrics)
        }

        logger.info("Comprehensive report generated")
        return report

    def _generate_insights(self, rewards: List[float], metrics: Dict[str, Any]) -> List[str]:
        """
        Generate human-readable insights from the data.

        This shows how to automatically interpret data and provide
        actionable feedback - a key skill in ML engineering.

        Args:
            rewards: Training rewards
            metrics: Learning metrics

        Returns:
            List of insight strings

        Educational Notes:
        - Automated insight generation saves time
        - Domain knowledge is crucial for meaningful interpretations
        - Clear, actionable language is important for stakeholders
        """
        insights = []

        if rewards:
            # Trend analysis
            first_half = rewards[:len(rewards)//2]
            second_half = rewards[len(rewards)//2:]

            if first_half and second_half:
                avg_first = sum(first_half) / len(first_half)
                avg_second = sum(second_half) / len(second_half)

                if avg_second > avg_first * 1.1:  # 10% improvement
                    insights.append("🎉 Significant learning progress detected! Performance improved by {:.1f}%".format(
                        ((avg_second - avg_first) / avg_first) * 100))
                elif avg_second < avg_first * 0.9:  # 10% decline
                    insights.append("⚠️  Performance declined. Consider adjusting learning parameters.")

        # Pattern analysis
        if 'patterns' in metrics and metrics['patterns']:
            top_pattern = max(metrics['patterns'].items(), key=lambda x: x[1])
            insights.append(f"📊 Most common prompt pattern: '{top_pattern[0]}' (appears {top_pattern[1]} times)")

        # Success rate analysis
        success_rate = metrics.get('success_rate', 0)
        if success_rate > 0.8:
            insights.append("✅ High success rate indicates good learning performance")
        elif success_rate < 0.5:
            insights.append("🔄 Low success rate suggests the model needs more training or parameter tuning")

        return insights if insights else ["📈 Analysis in progress - more data needed for insights"]