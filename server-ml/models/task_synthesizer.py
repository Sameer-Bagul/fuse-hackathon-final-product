class TaskSynthesizer:
    def synthesize(self, tasks):
        # Synthesize a new task by averaging rewards across tasks
        if not tasks:
            return []
        num_actions = len(tasks[0])
        return [sum(task[i] for task in tasks) / len(tasks) for i in range(num_actions)]