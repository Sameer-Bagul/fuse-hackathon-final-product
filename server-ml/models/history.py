import time
from .prompt import Prompt

class History:
    def __init__(self):
        self.interactions = []

    def add_interaction(self, prompt, response, reward=None, action=None, source="user"):
        self.interactions.append({
            'prompt': prompt,
            'response': response,
            'reward': reward,
            'action': action,
            'timestamp': time.time(),
            'source': source  # "user" or "ai"
        })

    def get_recent_interactions(self, n=10):
        return self.interactions[-n:]

    def get_success_rate(self):
        if not self.interactions:
            return 0
        successful = sum(1 for i in self.interactions if i['reward'] and i['reward'] > 0.5)
        return successful / len(self.interactions)

    def get_pattern_frequency(self):
        patterns = {}
        for interaction in self.interactions:
            keywords = interaction['prompt'].get_keywords()
            for kw in keywords:
                patterns[kw] = patterns.get(kw, 0) + 1
        return patterns

    def get_all_interactions_chronological(self):
        """Get all interactions sorted by timestamp (chronological order)"""
        return sorted(self.interactions, key=lambda x: x.get('timestamp', 0))