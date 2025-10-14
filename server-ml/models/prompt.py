class Prompt:
    def __init__(self, text, timestamp=None, metadata=None):
        self.text = text
        self.timestamp = timestamp or ""
        self.metadata = metadata or {}

    def get_keywords(self):
        return self.text.lower().split()

    def __str__(self):
        return f"Prompt: {self.text}"