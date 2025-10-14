import re
import logging
from typing import List, Dict, Set, Optional, Any, Tuple
from collections import defaultdict
from .curriculum import CurriculumTree, SkillNode, DifficultyLevel
from .prompt import Prompt

logger = logging.getLogger(__name__)

class CurriculumGenerator:
    """Analyzes user prompts and generates personalized learning curricula"""

    def __init__(self):
        self.skill_keywords = self._initialize_skill_keywords()
        self.topic_mappings = self._initialize_topic_mappings()
        self.difficulty_progression = self._initialize_difficulty_progression()

        # Try to import NLTK - make it optional
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            self.nltk_available = True
            # Try to download NLTK data if available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                except:
                    logger.warning("NLTK data not available, using basic tokenization")
        except ImportError:
            logger.warning("NLTK not available, using basic tokenization")
            self.nltk_available = False

    def _initialize_skill_keywords(self) -> Dict[str, List[str]]:
        """Initialize keyword mappings for different skills"""
        return {
            "python_basics": [
                "python", "programming", "syntax", "variables", "functions", "loops",
                "conditionals", "basic", "fundamentals", "beginner", "intro"
            ],
            "data_structures": [
                "list", "dictionary", "tuple", "set", "array", "data structure",
                "collection", "container", "stack", "queue", "linked list"
            ],
            "numpy_basics": [
                "numpy", "array", "matrix", "vector", "numerical", "scientific computing",
                "ndarray", "broadcasting", "indexing", "slicing"
            ],
            "pandas_basics": [
                "pandas", "dataframe", "series", "data manipulation", "csv", "excel",
                "data cleaning", "data analysis", "groupby", "merge", "join"
            ],
            "ml_concepts": [
                "machine learning", "ml", "supervised", "unsupervised", "training",
                "testing", "model", "algorithm", "prediction", "classification"
            ],
            "linear_regression": [
                "linear regression", "regression", "linear model", "least squares",
                "cost function", "gradient descent", "correlation", "prediction"
            ],
            "logistic_regression": [
                "logistic regression", "classification", "sigmoid", "binary classification",
                "cross entropy", "odds ratio", "probability"
            ],
            "decision_trees": [
                "decision tree", "tree", "random forest", "ensemble", "boosting",
                "feature importance", "splitting", "leaf", "node"
            ],
            "neural_networks": [
                "neural network", "deep learning", "perceptron", "backpropagation",
                "activation function", "layer", "hidden layer", "weights"
            ],
            "computer_vision": [
                "computer vision", "image", "cnn", "convolutional", "opencv",
                "object detection", "image processing", "pixel", "filter"
            ],
            "nlp_basics": [
                "natural language processing", "nlp", "text", "language model",
                "tokenization", "embedding", "transformer", "sentiment", "chatbot"
            ]
        }

    def _initialize_topic_mappings(self) -> Dict[str, List[str]]:
        """Initialize broader topic mappings"""
        return {
            "programming": ["python", "javascript", "java", "cpp", "coding", "software"],
            "data_science": ["data", "analysis", "statistics", "visualization", "dataset"],
            "machine_learning": ["ml", "ai", "artificial intelligence", "model", "algorithm"],
            "deep_learning": ["neural network", "deep", "cnn", "rnn", "transformer"],
            "computer_vision": ["image", "vision", "opencv", "detection", "recognition"],
            "nlp": ["language", "text", "sentiment", "translation", "generation"]
        }

    def _initialize_difficulty_progression(self) -> Dict[str, List[str]]:
        """Define skill progression paths for different topics"""
        return {
            "programming": [
                "python_basics", "data_structures", "numpy_basics", "pandas_basics"
            ],
            "machine_learning": [
                "ml_concepts", "linear_regression", "logistic_regression",
                "decision_trees", "neural_networks"
            ],
            "data_science": [
                "numpy_basics", "pandas_basics", "ml_concepts", "linear_regression"
            ],
            "deep_learning": [
                "ml_concepts", "neural_networks", "computer_vision", "nlp_basics"
            ]
        }

    def analyze_prompt(self, prompt: Prompt) -> Dict[str, Any]:
        """Analyze a user prompt and extract learning topics and skills"""
        text = prompt.text.lower()

        # Extract keywords
        keywords = self._extract_keywords(text)

        # Identify relevant skills
        relevant_skills = self._identify_relevant_skills(keywords)

        # Determine primary topics
        primary_topics = self._determine_primary_topics(keywords, relevant_skills)

        # Assess complexity level
        complexity_level = self._assess_complexity(text, keywords)

        return {
            'keywords': keywords,
            'relevant_skills': relevant_skills,
            'primary_topics': primary_topics,
            'complexity_level': complexity_level,
            'skill_confidence': self._calculate_skill_confidence(keywords, relevant_skills)
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        if self.nltk_available:
            try:
                # Try NLTK tokenization
                tokens = word_tokenize(text)
                stop_words = set(stopwords.words('english'))
            except:
                # Fallback to basic tokenization
                tokens = re.findall(r'\b\w+\b', text)
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        else:
            # Basic tokenization without NLTK
            tokens = re.findall(r'\b\w+\b', text)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        # Filter keywords
        keywords = []
        for token in tokens:
            token = token.lower()
            if (len(token) > 2 and
                token not in stop_words and
                not token.isdigit() and
                not re.match(r'^[^\w\s]+$', token)):  # Filter out punctuation-only tokens
                keywords.append(token)

        return list(set(keywords))  # Remove duplicates

    def _identify_relevant_skills(self, keywords: List[str]) -> List[Tuple[str, float]]:
        """Identify skills relevant to the keywords"""
        skill_scores = defaultdict(float)

        for keyword in keywords:
            for skill_id, skill_keywords in self.skill_keywords.items():
                if keyword in skill_keywords:
                    skill_scores[skill_id] += 1.0

                # Partial matching for compound terms
                for skill_kw in skill_keywords:
                    if skill_kw in keyword or keyword in skill_kw:
                        skill_scores[skill_id] += 0.5

        # Sort by score and return top skills
        sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)
        return [(skill, score) for skill, score in sorted_skills if score > 0]

    def _determine_primary_topics(self, keywords: List[str], relevant_skills: List[Tuple[str, float]]) -> List[str]:
        """Determine the primary learning topics"""
        topic_scores = defaultdict(float)

        # Score topics based on keywords
        for keyword in keywords:
            for topic, topic_keywords in self.topic_mappings.items():
                if keyword in topic_keywords:
                    topic_scores[topic] += 1.0

        # Score topics based on relevant skills
        skill_topic_mapping = {
            "python_basics": "programming", "data_structures": "programming",
            "numpy_basics": "data_science", "pandas_basics": "data_science",
            "ml_concepts": "machine_learning", "linear_regression": "machine_learning",
            "logistic_regression": "machine_learning", "decision_trees": "machine_learning",
            "neural_networks": "deep_learning", "computer_vision": "computer_vision",
            "nlp_basics": "nlp"
        }

        for skill_id, _ in relevant_skills:
            topic = skill_topic_mapping.get(skill_id)
            if topic:
                topic_scores[topic] += 2.0  # Skills weight more than keywords

        # Return top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, score in sorted_topics if score > 0][:3]  # Top 3 topics

    def _assess_complexity(self, text: str, keywords: List[str]) -> str:
        """Assess the complexity level of the prompt"""
        # Simple heuristic based on text length and advanced keywords
        advanced_keywords = {
            'neural', 'network', 'deep', 'learning', 'convolutional', 'recurrent',
            'transformer', 'attention', 'gradient', 'backpropagation', 'optimization',
            'regularization', 'ensemble', 'boosting', 'clustering', 'dimensionality'
        }

        text_length = len(text.split())
        advanced_count = sum(1 for kw in keywords if any(adv in kw for adv in advanced_keywords))

        if text_length < 10 and advanced_count == 0:
            return "basic"
        elif text_length < 20 and advanced_count <= 1:
            return "intermediate"
        elif advanced_count >= 2 or text_length > 30:
            return "advanced"
        else:
            return "intermediate"

    def _calculate_skill_confidence(self, keywords: List[str], relevant_skills: List[Tuple[str, float]]) -> Dict[str, float]:
        """Calculate confidence scores for identified skills"""
        confidence_scores = {}

        total_keywords = len(keywords)
        if total_keywords == 0:
            return confidence_scores

        for skill_id, score in relevant_skills:
            # Base confidence on keyword match ratio
            skill_keywords = self.skill_keywords.get(skill_id, [])
            matched_keywords = sum(1 for kw in keywords if kw in skill_keywords)
            keyword_confidence = matched_keywords / len(skill_keywords) if skill_keywords else 0

            # Combine with scoring confidence
            max_possible_score = len(skill_keywords) * 1.5  # Max score per keyword
            score_confidence = score / max_possible_score if max_possible_score > 0 else 0

            # Weighted average
            confidence_scores[skill_id] = (keyword_confidence * 0.6 + score_confidence * 0.4)

        return confidence_scores

    def generate_curriculum(self, prompt_analysis: Dict[str, Any], base_curriculum: Optional[CurriculumTree] = None) -> CurriculumTree:
        """Generate a personalized curriculum based on prompt analysis"""
        if base_curriculum is None:
            base_curriculum = CurriculumTree.create_default_curriculum()

        # Create new curriculum tree
        curriculum = CurriculumTree()

        # Get relevant skills from analysis
        relevant_skills = prompt_analysis.get('relevant_skills', [])
        primary_topics = prompt_analysis.get('primary_topics', [])
        complexity_level = prompt_analysis.get('complexity_level', 'intermediate')

        # Select and adapt skills based on analysis
        selected_skills = self._select_curriculum_skills(
            relevant_skills, primary_topics, complexity_level, base_curriculum
        )

        # Add selected skills to curriculum
        for skill_id in selected_skills:
            if skill_id in base_curriculum.skills:
                base_skill = base_curriculum.skills[skill_id]
                curriculum.add_skill(base_skill)

        # Add prerequisite relationships
        self._add_prerequisites(curriculum, selected_skills, base_curriculum)

        # Validate the generated curriculum
        errors = curriculum.validate_curriculum()
        if errors:
            logger.warning(f"Curriculum validation errors: {errors}")

        return curriculum

    def _select_curriculum_skills(self, relevant_skills: List[Tuple[str, float]],
                                primary_topics: List[str], complexity_level: str,
                                base_curriculum: CurriculumTree) -> List[str]:
        """Select skills for the curriculum based on analysis"""
        selected_skills = set()

        # Always include highly relevant skills
        for skill_id, score in relevant_skills:
            if score >= 1.0:  # High confidence threshold
                selected_skills.add(skill_id)

        # Add skills from primary topics
        for topic in primary_topics:
            if topic in self.difficulty_progression:
                progression = self.difficulty_progression[topic]

                # Select skills based on complexity level
                if complexity_level == "basic":
                    selected_skills.update(progression[:2])  # First 2 skills
                elif complexity_level == "intermediate":
                    selected_skills.update(progression[:4])  # First 4 skills
                else:  # advanced
                    selected_skills.update(progression)  # All skills

        # Ensure we have at least some basic skills
        if not selected_skills:
            basic_skills = ["python_basics", "ml_concepts", "numpy_basics"]
            selected_skills.update(basic_skills)

        # Convert to list and ensure all skills exist in base curriculum
        final_skills = [skill for skill in selected_skills if skill in base_curriculum.skills]

        return final_skills

    def _add_prerequisites(self, curriculum: CurriculumTree, selected_skills: List[str],
                          base_curriculum: CurriculumTree):
        """Add prerequisite skills to ensure curriculum completeness"""
        skills_to_add = set(selected_skills)
        added_skills = set()

        # Iteratively add prerequisites
        while skills_to_add:
            current_skill = skills_to_add.pop()

            if current_skill in added_skills:
                continue

            if current_skill in base_curriculum.skills:
                base_skill = base_curriculum.skills[current_skill]

                # Add prerequisites first
                for prereq in base_skill.prerequisites:
                    if prereq not in added_skills and prereq not in skills_to_add:
                        skills_to_add.add(prereq)

                # Add the skill itself
                curriculum.add_skill(base_skill)
                added_skills.add(current_skill)

    def generate_progressive_tasks(self, curriculum: CurriculumTree,
                                 learner_progress: Optional[Any] = None,
                                 num_tasks: int = 5) -> List[Dict[str, Any]]:
        """Generate a series of tasks with increasing difficulty"""
        tasks = []

        # Get available skills in order
        available_skills = list(curriculum.skills.keys())

        if learner_progress:
            # Filter based on learner progress
            available_skills = learner_progress.get_available_skills(curriculum)

        if not available_skills:
            # Fallback to all skills
            available_skills = list(curriculum.skills.keys())

        # Sort skills by difficulty
        skill_difficulty_order = {
            DifficultyLevel.EASY: 0,
            DifficultyLevel.MEDIUM: 1,
            DifficultyLevel.HARD: 2,
            DifficultyLevel.EXPERT: 3
        }

        available_skills.sort(key=lambda s: skill_difficulty_order.get(
            curriculum.skills[s].difficulty, 0
        ))

        # Generate tasks with progressive difficulty
        for i in range(min(num_tasks, len(available_skills))):
            skill_id = available_skills[i]
            skill = curriculum.skills[skill_id]

            # Create task with increasing complexity
            task = self._create_progressive_task(skill, i, num_tasks)

            tasks.append({
                'task_id': f"{skill_id}_task_{i+1}",
                'skill_id': skill_id,
                'skill_name': skill.name,
                'difficulty': skill.difficulty.value,
                'task_description': task['description'],
                'expected_complexity': task['complexity'],
                'learning_objectives': skill.learning_objectives,
                'estimated_time': skill.estimated_time
            })

        return tasks

    def _create_progressive_task(self, skill: SkillNode, task_index: int, total_tasks: int) -> Dict[str, Any]:
        """Create a task with progressive complexity"""
        # Base complexity increases with task index
        base_complexity = task_index / max(1, total_tasks - 1)  # 0.0 to 1.0

        # Skill-specific task templates
        task_templates = {
            "python_basics": [
                "Write a Python function to {task}",
                "Create a Python script that {task}",
                "Implement a Python program for {task}"
            ],
            "data_structures": [
                "Implement a {structure} data structure in Python",
                "Create functions to manipulate {structure} operations",
                "Solve a problem using {structure} data structures"
            ],
            "numpy_basics": [
                "Use NumPy to perform {operation} on arrays",
                "Create and manipulate NumPy arrays for {task}",
                "Implement numerical operations using NumPy"
            ],
            "pandas_basics": [
                "Use Pandas to {operation} on DataFrames",
                "Perform data analysis tasks with Pandas",
                "Clean and manipulate data using Pandas operations"
            ],
            "ml_concepts": [
                "Explain the concept of {concept} in machine learning",
                "Describe how {algorithm} works",
                "Compare different approaches to {task}"
            ]
        }

        templates = task_templates.get(skill.skill_id, [
            f"Apply {skill.name.lower()} to solve a problem",
            f"Implement {skill.name.lower()} concepts",
            f"Demonstrate understanding of {skill.name.lower()}"
        ])

        # Select template based on complexity
        template_index = min(int(base_complexity * len(templates)), len(templates) - 1)
        template = templates[template_index]

        # Add complexity modifiers
        complexity_descriptors = {
            0: "basic",
            1: "intermediate",
            2: "advanced",
            3: "complex"
        }

        complexity_level = min(int(base_complexity * 4), 3)
        complexity_modifier = complexity_descriptors[complexity_level]

        description = f"{complexity_modifier.capitalize()} task: {template}"

        return {
            'description': description,
            'complexity': base_complexity,
            'complexity_level': complexity_modifier
        }