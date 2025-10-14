import matplotlib.pyplot as plt
import json

class Visualizer:
    def plot(self, data, title="Training Rewards"):
        n = len(data)

        # Create subplots: 2x2 grid
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=16)

        # Subplot 1: Learning curves (reward over episodes)
        axs[0, 0].plot(data)
        axs[0, 0].set_title("Learning Curves")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Reward")

        # Subplot 2: Before/After performance comparison
        half = n // 2
        avg_before = sum(data[:half]) / half if half > 0 else 0
        avg_after = sum(data[half:]) / (n - half) if n - half > 0 else 0
        axs[0, 1].bar(['Before (First Half)', 'After (Second Half)'], [avg_before, avg_after])
        axs[0, 1].set_title("Before/After Performance Comparison")
        axs[0, 1].set_ylabel("Average Reward")

        # Subplot 3: Difficulty progression over time (assuming linear increase for curriculum learning)
        difficulty = [i / (n - 1) if n > 1 else 0 for i in range(n)]
        axs[1, 0].plot(difficulty)
        axs[1, 0].set_title("Difficulty Progression")
        axs[1, 0].set_xlabel("Episode")
        axs[1, 0].set_ylabel("Difficulty Level")

        # Subplot 4: Smoothed learning curve (moving average)
        window_size = max(1, n // 50)  # Adjust window size based on data length
        smoothed = []
        for i in range(n):
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2 + 1)
            smoothed.append(sum(data[start:end]) / (end - start))
        axs[1, 1].plot(smoothed)
        axs[1, 1].set_title("Smoothed Learning Curve")
        axs[1, 1].set_xlabel("Episode")
        axs[1, 1].set_ylabel("Smoothed Reward")

        plt.tight_layout()
        plt.show()

    def plot_learning_progress(self, metrics, title="LLM Learning Progress"):
        # Plot for prompt patterns, response improvements, task generation metrics
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)

        # Subplot 1: Success rate over time
        if 'success_rates' in metrics:
            axs[0].plot(metrics['success_rates'])
            axs[0].set_title("Success Rate Over Time")
            axs[0].set_xlabel("Interaction")
            axs[0].set_ylabel("Success Rate")

        # Subplot 2: Pattern frequency (top 10)
        if 'patterns' in metrics:
            patterns = metrics['patterns']
            top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            words, freqs = zip(*top_patterns)
            axs[1].bar(words, freqs)
            axs[1].set_title("Top Prompt Patterns")
            axs[1].set_ylabel("Frequency")
            axs[1].tick_params(axis='x', rotation=45)

        # Subplot 3: Task generation metrics
        if 'task_metrics' in metrics:
            axs[2].plot(metrics['task_metrics'])
            axs[2].set_title("Task Generation Metrics")
            axs[2].set_xlabel("Task")
            axs[2].set_ylabel("Metric Value")

        plt.tight_layout()
        plt.show()

    def get_chartjs_data(self, data, metrics=None):
        # Prepare data for Chart.js
        chart_data = {
            'learning_curve': {
                'labels': list(range(len(data))),
                'datasets': [{
                    'label': 'Reward',
                    'data': data,
                    'borderColor': 'rgb(75, 192, 192)',
                    'tension': 0.1
                }]
            }
        }
        if metrics:
            if 'success_rates' in metrics:
                chart_data['success_rate'] = {
                    'labels': list(range(len(metrics['success_rates']))),
                    'datasets': [{
                        'label': 'Success Rate',
                        'data': metrics['success_rates'],
                        'borderColor': 'rgb(255, 99, 132)',
                        'tension': 0.1
                    }]
                }
            if 'patterns' in metrics:
                top_patterns = sorted(metrics['patterns'].items(), key=lambda x: x[1], reverse=True)[:10]
                words, freqs = zip(*top_patterns)
                chart_data['patterns'] = {
                    'labels': words,
                    'datasets': [{
                        'label': 'Frequency',
                        'data': freqs,
                        'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                        'borderColor': 'rgba(54, 162, 235, 1)',
                        'borderWidth': 1
                    }]
                }
        return json.dumps(chart_data)