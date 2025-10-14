from typing import List, Dict, Set, Optional, Any
from enum import Enum
import json
import time
from dataclasses import dataclass, asdict

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

@dataclass
class SkillNode:
    """Individual skill with prerequisites and difficulty"""
    skill_id: str
    name: str
    description: str
    difficulty: DifficultyLevel
    prerequisites: List[str]  # List of skill_ids that must be mastered first
    estimated_time: int  # Estimated time in minutes to master
    category: str  # e.g., "machine_learning", "data_science", "programming"
    learning_objectives: List[str]  # What the learner should know after mastering
    mastery_threshold: float = 0.8  # Score needed to consider skill mastered (0-1)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['difficulty'] = self.difficulty.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SkillNode':
        data['difficulty'] = DifficultyLevel(data['difficulty'])
        return cls(**data)

@dataclass
class CurriculumProgress:
    """Tracks learner advancement through the curriculum"""
    learner_id: str
    skill_progress: Dict[str, Dict[str, Any]]  # skill_id -> progress data
    completed_skills: Set[str]
    current_difficulty: DifficultyLevel
    total_time_spent: int  # Total time spent learning in minutes
    last_updated: float

    def __init__(self, learner_id: str):
        self.learner_id = learner_id
        self.skill_progress = {}
        self.completed_skills = set()
        self.current_difficulty = DifficultyLevel.EASY
        self.total_time_spent = 0
        self.last_updated = time.time()

    def update_skill_progress(self, skill_id: str, score: float, time_spent: int = 0):
        """Update progress for a specific skill"""
        if skill_id not in self.skill_progress:
            self.skill_progress[skill_id] = {
                'attempts': 0,
                'best_score': 0.0,
                'total_score': 0.0,
                'time_spent': 0,
                'last_attempt': 0.0,
                'mastered': False
            }

        progress = self.skill_progress[skill_id]
        progress['attempts'] += 1
        progress['total_score'] += score
        progress['best_score'] = max(progress['best_score'], score)
        progress['time_spent'] += time_spent
        progress['last_attempt'] = time.time()

        # Check if skill is mastered
        if score >= 0.8 and not progress['mastered']:  # Using default threshold
            progress['mastered'] = True
            self.completed_skills.add(skill_id)

        self.last_updated = time.time()
        self.total_time_spent += time_spent

    def get_skill_mastery_level(self, skill_id: str) -> float:
        """Get mastery level for a skill (0-1)"""
        if skill_id not in self.skill_progress:
            return 0.0
        return self.skill_progress[skill_id]['best_score']

    def is_skill_mastered(self, skill_id: str) -> bool:
        """Check if a skill is mastered"""
        return skill_id in self.completed_skills

    def get_available_skills(self, curriculum_tree: 'CurriculumTree') -> List[str]:
        """Get skills that are available to learn (prerequisites met)"""
        available = []
        for skill_id, skill in curriculum_tree.skills.items():
            if self.is_skill_mastered(skill_id):
                continue
            # Check if all prerequisites are met
            prereqs_met = all(self.is_skill_mastered(prereq) for prereq in skill.prerequisites)
            if prereqs_met:
                available.append(skill_id)
        return available

    def get_recommended_difficulty(self) -> DifficultyLevel:
        """Determine recommended difficulty based on progress"""
        completion_rate = len(self.completed_skills) / max(1, len(self.skill_progress))

        if completion_rate < 0.3:
            return DifficultyLevel.EASY
        elif completion_rate < 0.6:
            return DifficultyLevel.MEDIUM
        elif completion_rate < 0.8:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.EXPERT

    def to_dict(self) -> Dict[str, Any]:
        return {
            'learner_id': self.learner_id,
            'skill_progress': self.skill_progress,
            'completed_skills': list(self.completed_skills),
            'current_difficulty': self.current_difficulty.value,
            'total_time_spent': self.total_time_spent,
            'last_updated': self.last_updated
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurriculumProgress':
        progress = cls(data['learner_id'])
        progress.skill_progress = data['skill_progress']
        progress.completed_skills = set(data['completed_skills'])
        progress.current_difficulty = DifficultyLevel(data['current_difficulty'])
        progress.total_time_spent = data['total_time_spent']
        progress.last_updated = data['last_updated']
        return progress

class CurriculumTree:
    """Complete skill progression structure"""

    def __init__(self):
        self.skills: Dict[str, SkillNode] = {}
        self.categories: Dict[str, List[str]] = {}  # category -> list of skill_ids
        self.difficulty_levels: Dict[DifficultyLevel, List[str]] = {}  # difficulty -> list of skill_ids

    def add_skill(self, skill: SkillNode):
        """Add a skill to the curriculum tree"""
        self.skills[skill.skill_id] = skill

        # Update categories
        if skill.category not in self.categories:
            self.categories[skill.category] = []
        self.categories[skill.category].append(skill.skill_id)

        # Update difficulty levels
        if skill.difficulty not in self.difficulty_levels:
            self.difficulty_levels[skill.difficulty] = []
        self.difficulty_levels[skill.difficulty].append(skill.skill_id)

    def get_skill_chain(self, skill_id: str) -> List[str]:
        """Get the full prerequisite chain for a skill"""
        if skill_id not in self.skills:
            return []

        chain = []
        visited = set()

        def build_chain(current_skill_id: str):
            if current_skill_id in visited:
                return
            visited.add(current_skill_id)

            skill = self.skills[current_skill_id]
            for prereq in skill.prerequisites:
                build_chain(prereq)

            chain.append(current_skill_id)

        build_chain(skill_id)
        return chain

    def get_next_skills(self, completed_skills: Set[str]) -> List[str]:
        """Get skills that can be learned next given completed skills"""
        next_skills = []
        for skill_id, skill in self.skills.items():
            if skill_id in completed_skills:
                continue
            # Check if all prerequisites are completed
            if all(prereq in completed_skills for prereq in skill.prerequisites):
                next_skills.append(skill_id)
        return next_skills

    def get_skills_by_difficulty(self, difficulty: DifficultyLevel) -> List[str]:
        """Get all skills of a specific difficulty level"""
        return self.difficulty_levels.get(difficulty, [])

    def get_skills_by_category(self, category: str) -> List[str]:
        """Get all skills in a specific category"""
        return self.categories.get(category, [])

    def validate_curriculum(self) -> List[str]:
        """Validate the curriculum for consistency"""
        errors = []

        # Check for circular dependencies
        for skill_id in self.skills:
            chain = self.get_skill_chain(skill_id)
            if len(chain) != len(set(chain)):
                errors.append(f"Circular dependency detected involving skill {skill_id}")

        # Check that all prerequisites exist
        for skill_id, skill in self.skills.items():
            for prereq in skill.prerequisites:
                if prereq not in self.skills:
                    errors.append(f"Skill {skill_id} references non-existent prerequisite {prereq}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            'skills': {skill_id: skill.to_dict() for skill_id, skill in self.skills.items()},
            'categories': self.categories,
            'difficulty_levels': {diff.value: skills for diff, skills in self.difficulty_levels.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurriculumTree':
        tree = cls()
        for skill_id, skill_data in data['skills'].items():
            tree.add_skill(SkillNode.from_dict(skill_data))
        return tree

    @classmethod
    def create_default_curriculum(cls) -> 'CurriculumTree':
        """Create a default ML curriculum tree"""
        tree = cls()

        # Basic Programming Skills
        tree.add_skill(SkillNode(
            skill_id="python_basics",
            name="Python Basics",
            description="Fundamental Python programming concepts",
            difficulty=DifficultyLevel.EASY,
            prerequisites=[],
            estimated_time=120,
            category="programming",
            learning_objectives=["Basic syntax", "Data types", "Control structures", "Functions"]
        ))

        tree.add_skill(SkillNode(
            skill_id="data_structures",
            name="Data Structures",
            description="Lists, dictionaries, sets, and tuples",
            difficulty=DifficultyLevel.EASY,
            prerequisites=["python_basics"],
            estimated_time=90,
            category="programming",
            learning_objectives=["List operations", "Dictionary usage", "Set operations", "Tuple handling"]
        ))

        # Data Science Fundamentals
        tree.add_skill(SkillNode(
            skill_id="numpy_basics",
            name="NumPy Fundamentals",
            description="Array operations and mathematical computing",
            difficulty=DifficultyLevel.EASY,
            prerequisites=["python_basics"],
            estimated_time=60,
            category="data_science",
            learning_objectives=["Array creation", "Indexing and slicing", "Mathematical operations", "Broadcasting"]
        ))

        tree.add_skill(SkillNode(
            skill_id="pandas_basics",
            name="Pandas Fundamentals",
            description="Data manipulation and analysis",
            difficulty=DifficultyLevel.MEDIUM,
            prerequisites=["data_structures", "numpy_basics"],
            estimated_time=120,
            category="data_science",
            learning_objectives=["DataFrame operations", "Series manipulation", "Data cleaning", "Basic statistics"]
        ))

        # Machine Learning Basics
        tree.add_skill(SkillNode(
            skill_id="ml_concepts",
            name="ML Concepts",
            description="Core machine learning concepts and terminology",
            difficulty=DifficultyLevel.EASY,
            prerequisites=[],
            estimated_time=90,
            category="machine_learning",
            learning_objectives=["Supervised vs unsupervised learning", "Training and testing", "Overfitting", "Model evaluation"]
        ))

        tree.add_skill(SkillNode(
            skill_id="linear_regression",
            name="Linear Regression",
            description="Simple linear and multiple regression",
            difficulty=DifficultyLevel.MEDIUM,
            prerequisites=["ml_concepts", "numpy_basics", "pandas_basics"],
            estimated_time=90,
            category="machine_learning",
            learning_objectives=["Simple linear regression", "Multiple regression", "Cost functions", "Gradient descent"]
        ))

        tree.add_skill(SkillNode(
            skill_id="logistic_regression",
            name="Logistic Regression",
            description="Binary and multiclass classification",
            difficulty=DifficultyLevel.MEDIUM,
            prerequisites=["linear_regression"],
            estimated_time=90,
            category="machine_learning",
            learning_objectives=["Sigmoid function", "Binary classification", "Multiclass classification", "Regularization"]
        ))

        # Advanced Topics
        tree.add_skill(SkillNode(
            skill_id="decision_trees",
            name="Decision Trees",
            description="Tree-based models and ensemble methods",
            difficulty=DifficultyLevel.MEDIUM,
            prerequisites=["ml_concepts", "pandas_basics"],
            estimated_time=120,
            category="machine_learning",
            learning_objectives=["Decision tree algorithm", "Random forests", "Gradient boosting", "Feature importance"]
        ))

        tree.add_skill(SkillNode(
            skill_id="neural_networks",
            name="Neural Networks",
            description="Deep learning fundamentals",
            difficulty=DifficultyLevel.HARD,
            prerequisites=["logistic_regression", "numpy_basics"],
            estimated_time=180,
            category="machine_learning",
            learning_objectives=["Perceptrons", "Backpropagation", "Activation functions", "Network architecture"]
        ))

        tree.add_skill(SkillNode(
            skill_id="computer_vision",
            name="Computer Vision",
            description="Image processing and CNNs",
            difficulty=DifficultyLevel.HARD,
            prerequisites=["neural_networks"],
            estimated_time=240,
            category="machine_learning",
            learning_objectives=["Convolutional networks", "Image preprocessing", "Object detection", "Transfer learning"]
        ))

        tree.add_skill(SkillNode(
            skill_id="nlp_basics",
            name="Natural Language Processing",
            description="Text processing and language models",
            difficulty=DifficultyLevel.HARD,
            prerequisites=["neural_networks", "pandas_basics"],
            estimated_time=240,
            category="machine_learning",
            learning_objectives=["Text preprocessing", "Word embeddings", "Sequence models", "Transformers"]
        ))

        return tree