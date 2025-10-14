import random
import logging
from typing import List, Dict, Optional, Any, Tuple
from .prompt import Prompt
from .curriculum import CurriculumTree, CurriculumProgress, DifficultyLevel, SkillNode

logger = logging.getLogger(__name__)

class TaskGenerator:
    def __init__(self, num_actions=10, curriculum_tree=None):
        self.num_actions = num_actions
        self.generated_tasks = []
        self.curriculum_tree = curriculum_tree or CurriculumTree.create_default_curriculum()
        self.difficulty_scaling = {
            DifficultyLevel.EASY: 1.0,
            DifficultyLevel.MEDIUM: 1.5,
            DifficultyLevel.HARD: 2.0,
            DifficultyLevel.EXPERT: 2.5
        }
        self.skill_focus_weights = {}  # skill_id -> weight for task generation

    def set_curriculum_tree(self, curriculum_tree: CurriculumTree):
        """Set the curriculum tree for task generation"""
        self.curriculum_tree = curriculum_tree

    def update_skill_focus(self, skill_id: str, weight: float):
        """Update focus weight for a specific skill"""
        self.skill_focus_weights[skill_id] = weight

    def generate_task(self, prompt=None, learner_progress: Optional[CurriculumProgress] = None,
                      target_difficulty: Optional[DifficultyLevel] = None) -> List[float]:
        """
        Generate a curriculum-aware task with difficulty scaling and skill progression

        Args:
            prompt: Optional prompt for context
            learner_progress: Current learner progress for personalization
            target_difficulty: Target difficulty level for the task
        """
        # Determine target difficulty
        if target_difficulty is None:
            if learner_progress:
                target_difficulty = learner_progress.get_recommended_difficulty()
            else:
                target_difficulty = DifficultyLevel.EASY

        # Get available skills for the learner
        available_skills = []
        if learner_progress:
            available_skills = learner_progress.get_available_skills(self.curriculum_tree)
        else:
            # If no progress, start with skills that have no prerequisites
            available_skills = [skill_id for skill_id, skill in self.curriculum_tree.skills.items()
                              if not skill.prerequisites]

        if not available_skills:
            # Fallback to all skills if no available ones
            available_skills = list(self.curriculum_tree.skills.keys())

        # Select target skill based on focus weights and availability
        target_skill_id = self._select_target_skill(available_skills, learner_progress)

        # Generate task based on prompt and curriculum context
        if prompt:
            task = self._generate_curriculum_task(prompt, target_skill_id, target_difficulty, learner_progress)
        else:
            task = self._generate_default_curriculum_task(target_skill_id, target_difficulty, learner_progress)

        # Apply progressive difficulty scaling
        task = self._apply_progressive_scaling(task, target_skill_id, target_difficulty, learner_progress)

        self.generated_tasks.append({
            'task': task,
            'target_skill': target_skill_id,
            'difficulty': target_difficulty.value,
            'timestamp': random.random()  # For tracking
        })

        return task

    def _select_target_skill(self, available_skills: List[str],
                           learner_progress: Optional[CurriculumProgress]) -> str:
        """Select which skill to focus on for task generation"""
        if not available_skills:
            return random.choice(list(self.curriculum_tree.skills.keys()))

        # Weight skills by focus and learner progress
        skill_weights = {}
        for skill_id in available_skills:
            weight = 1.0

            # Apply focus weights
            if skill_id in self.skill_focus_weights:
                weight *= self.skill_focus_weights[skill_id]

            # Prefer skills the learner hasn't mastered yet
            if learner_progress and learner_progress.is_skill_mastered(skill_id):
                weight *= 0.3  # Reduce weight for already mastered skills
            else:
                weight *= 1.5  # Increase weight for unmastered skills

            # Consider skill difficulty vs learner level
            if learner_progress:
                skill = self.curriculum_tree.skills[skill_id]
                learner_level = learner_progress.get_recommended_difficulty()
                if skill.difficulty == learner_level:
                    weight *= 2.0  # Prefer skills at learner's current level
                elif skill.difficulty.value > learner_level.value:
                    weight *= 0.5  # Reduce weight for too difficult skills

            skill_weights[skill_id] = weight

        # Select skill based on weights
        total_weight = sum(skill_weights.values())
        if total_weight == 0:
            return random.choice(available_skills)

        pick = random.uniform(0, total_weight)
        current_weight = 0
        for skill_id, weight in skill_weights.items():
            current_weight += weight
            if pick <= current_weight:
                return skill_id

        return available_skills[0]  # Fallback

    def _generate_curriculum_task(self, prompt: Prompt, target_skill_id: str,
                                target_difficulty: DifficultyLevel,
                                learner_progress: Optional[CurriculumProgress]) -> List[float]:
        """Generate task based on prompt and curriculum context"""
        keywords = prompt.get_keywords()
        skill = self.curriculum_tree.skills[target_skill_id]

        # Base priorities from keywords
        priorities = [1.0] * self.num_actions

        # Boost actions related to target skill
        skill_boost = self._get_skill_action_mapping(target_skill_id)
        for action_idx in skill_boost:
            if action_idx < self.num_actions:
                priorities[action_idx] += 1.0

        # Boost actions related to prerequisites (reinforcement)
        if learner_progress:
            for prereq_id in skill.prerequisites:
                if learner_progress.is_skill_mastered(prereq_id):
                    prereq_boost = self._get_skill_action_mapping(prereq_id)
                    for action_idx in prereq_boost:
                        if action_idx < self.num_actions:
                            priorities[action_idx] += 0.5

        # Keyword-based boosts
        for i, kw in enumerate(keywords):
            if i < self.num_actions:
                priorities[i] += 0.3

        # Add controlled randomness based on difficulty
        randomness_scale = 0.5 / self.difficulty_scaling[target_difficulty]  # Less random for harder tasks
        task = [p + random.random() * randomness_scale for p in priorities]

        return task

    def _generate_default_curriculum_task(self, target_skill_id: str,
                                        target_difficulty: DifficultyLevel,
                                        learner_progress: Optional[CurriculumProgress]) -> List[float]:
        """Generate default task focused on curriculum skill"""
        skill = self.curriculum_tree.skills[target_skill_id]

        # Base task with skill-specific boosts
        task = [random.random() for _ in range(self.num_actions)]

        # Apply skill-specific action boosts
        skill_boost = self._get_skill_action_mapping(target_skill_id)
        for action_idx in skill_boost:
            if action_idx < self.num_actions:
                task[action_idx] += 1.0

        # Apply difficulty scaling to randomness
        scaling_factor = self.difficulty_scaling[target_difficulty]
        task = [t * scaling_factor for t in task]

        return task

    def _get_skill_action_mapping(self, skill_id: str) -> List[int]:
        """Map skills to action indices for task generation"""
        # Simple hash-based mapping - in practice, this could be more sophisticated
        skill_actions = {
            "python_basics": [0, 1, 2],
            "data_structures": [1, 2, 3],
            "numpy_basics": [2, 3, 4],
            "pandas_basics": [3, 4, 5],
            "ml_concepts": [4, 5, 6],
            "linear_regression": [5, 6, 7],
            "logistic_regression": [6, 7, 8],
            "decision_trees": [7, 8, 9],
            "neural_networks": [8, 9, 0],
            "computer_vision": [9, 0, 1],
            "nlp_basics": [0, 1, 2]
        }

        return skill_actions.get(skill_id, [random.randint(0, self.num_actions-1) for _ in range(3)])

    def prioritize_tasks(self, tasks: List[List[float]],
                        learner_progress: Optional[CurriculumProgress] = None) -> List[List[float]]:
        """
        Prioritize tasks based on curriculum progress and learner needs

        Args:
            tasks: List of task vectors
            learner_progress: Current learner progress for personalization
        """
        if not tasks:
            return tasks

        # Enhanced prioritization considering curriculum
        task_scores = []

        for i, task in enumerate(tasks):
            score = sum(task) / len(task)  # Base score

            # Get task metadata if available
            task_meta = None
            if i < len(self.generated_tasks) and isinstance(self.generated_tasks[i], dict):
                task_meta = self.generated_tasks[i]

            if task_meta and learner_progress:
                skill_id = task_meta.get('target_skill')
                difficulty = task_meta.get('difficulty')

                if skill_id and skill_id in self.curriculum_tree.skills:
                    skill = self.curriculum_tree.skills[skill_id]

                    # Boost tasks for unmastered skills
                    if not learner_progress.is_skill_mastered(skill_id):
                        score += 0.3

                    # Boost tasks at appropriate difficulty
                    recommended_diff = learner_progress.get_recommended_difficulty()
                    if difficulty == recommended_diff.value:
                        score += 0.2

                    # Boost tasks for skills with unmet prerequisites
                    available_skills = learner_progress.get_available_skills(self.curriculum_tree)
                    if skill_id in available_skills:
                        score += 0.1

            task_scores.append((score, i))

        # Sort by score descending
        task_scores.sort(reverse=True)
        prioritized_tasks = [tasks[i] for _, i in task_scores]

        return prioritized_tasks

    def adjust_curriculum_based_on_performance(self, learner_progress: CurriculumProgress,
                                              recent_performance: List[float],
                                              feedback_adjustments: Optional[Dict[str, Any]] = None):
        """
        Dynamically adjust curriculum difficulty and focus based on learner performance

        Args:
            learner_progress: Current learner progress
            recent_performance: List of recent task performance scores
            feedback_adjustments: Optional feedback-based adjustments
        """
        if not recent_performance:
            return

        avg_performance = sum(recent_performance) / len(recent_performance)

        # Adjust difficulty based on performance
        if avg_performance > 0.8:
            # Performing well, can increase difficulty
            current_level = learner_progress.current_difficulty
            if current_level != DifficultyLevel.EXPERT:
                next_level = DifficultyLevel(current_level.value + 1)
                learner_progress.current_difficulty = next_level
                logger.info(f"Increased difficulty to {next_level.value}")
        elif avg_performance < 0.4:
            # Struggling, decrease difficulty
            current_level = learner_progress.current_difficulty
            if current_level != DifficultyLevel.EASY:
                prev_level = DifficultyLevel(current_level.value - 1)
                learner_progress.current_difficulty = prev_level
                logger.info(f"Decreased difficulty to {prev_level.value}")

        # Identify skill gaps and adjust focus
        available_skills = learner_progress.get_available_skills(self.curriculum_tree)
        struggling_skills = []

        for skill_id in available_skills:
            mastery = learner_progress.get_skill_mastery_level(skill_id)
            if mastery < 0.5:  # Struggling with this skill
                struggling_skills.append(skill_id)

        # Increase focus on struggling skills
        for skill_id in struggling_skills:
            self.skill_focus_weights[skill_id] = self.skill_focus_weights.get(skill_id, 1.0) * 1.5

        # Decrease focus on mastered skills
        mastered_skills = [s for s in available_skills if learner_progress.is_skill_mastered(s)]
        for skill_id in mastered_skills:
            self.skill_focus_weights[skill_id] = self.skill_focus_weights.get(skill_id, 1.0) * 0.8

    def _apply_progressive_scaling(self, task: List[float], target_skill_id: str,
                                 target_difficulty: DifficultyLevel,
                                 learner_progress: Optional[CurriculumProgress]) -> List[float]:
        """Apply progressive difficulty scaling based on curriculum and learner progress"""
        # Base difficulty scaling
        scaling_factor = self.difficulty_scaling[target_difficulty]
        scaled_task = [min(1.0, t * scaling_factor) for t in task]

        if not learner_progress:
            return scaled_task

        # Get skill-specific scaling based on mastery level
        mastery_level = learner_progress.get_skill_mastery_level(target_skill_id)

        # Progressive scaling: adjust based on current mastery
        if mastery_level < 0.3:
            # Beginner: reduce difficulty slightly to build confidence
            progressive_factor = 0.8
        elif mastery_level < 0.6:
            # Intermediate: standard difficulty
            progressive_factor = 1.0
        elif mastery_level < 0.8:
            # Advanced: increase difficulty to challenge
            progressive_factor = 1.2
        else:
            # Expert: significantly increase difficulty
            progressive_factor = 1.5

        # Apply progressive scaling
        scaled_task = [min(1.0, t * progressive_factor) for t in scaled_task]

        # Add curriculum-aware adjustments
        skill = self.curriculum_tree.skills.get(target_skill_id)
        if skill:
            # Adjust based on prerequisites mastery
            prereq_mastery = 0
            if skill.prerequisites:
                prereq_mastery = sum(learner_progress.get_skill_mastery_level(prereq)
                                    for prereq in skill.prerequisites) / len(skill.prerequisites)
            else:
                prereq_mastery = 1.0  # No prerequisites = full mastery

            # If prerequisites are not well mastered, reduce difficulty
            if prereq_mastery < 0.7:
                prereq_adjustment = 0.9
                scaled_task = [min(1.0, t * prereq_adjustment) for t in scaled_task]

        return scaled_task

    def generate_progressive_task_sequence(self, user_prompt: str, num_tasks: int = 5,
                                         learner_progress: Optional[CurriculumProgress] = None) -> List[Dict[str, Any]]:
        """Generate a sequence of tasks with increasing difficulty from a user prompt"""
        from .curriculum_generator import CurriculumGenerator

        # Analyze prompt and generate curriculum
        curriculum_gen = CurriculumGenerator()
        prompt = Prompt(user_prompt)
        analysis = curriculum_gen.analyze_prompt(prompt)
        curriculum = curriculum_gen.generate_curriculum(analysis)

        # Update our curriculum tree
        self.set_curriculum_tree(curriculum)

        # Generate progressive tasks
        progressive_tasks = curriculum_gen.generate_progressive_tasks(
            curriculum, learner_progress, num_tasks
        )

        # Convert to task vectors
        task_vectors = []
        for task_info in progressive_tasks:
            # Create task vector based on skill and difficulty
            skill_id = task_info['skill_id']
            difficulty = DifficultyLevel(task_info['difficulty'])

            # Generate base task for this skill
            base_task = self._generate_default_curriculum_task(skill_id, difficulty, learner_progress)

            # Apply progressive scaling
            scaled_task = self._apply_progressive_scaling(base_task, skill_id, difficulty, learner_progress)

            task_vectors.append({
                'task_vector': scaled_task,
                'task_info': task_info,
                'sequence_position': len(task_vectors) + 1
            })

        return task_vectors

    def get_curriculum_recommendations(self, learner_progress: CurriculumProgress) -> Dict[str, Any]:
        """Get curriculum-based recommendations for the learner"""
        available_skills = learner_progress.get_available_skills(self.curriculum_tree)
        recommended_difficulty = learner_progress.get_recommended_difficulty()

        recommendations = {
            'next_skills': available_skills[:3],  # Top 3 recommended skills
            'recommended_difficulty': recommended_difficulty.value,
            'skill_gaps': [],
            'progress_summary': {
                'completed_skills': len(learner_progress.completed_skills),
                'total_skills': len(self.curriculum_tree.skills),
                'completion_rate': len(learner_progress.completed_skills) / len(self.curriculum_tree.skills)
            }
        }

        # Identify skill gaps
        for skill_id in available_skills:
            mastery = learner_progress.get_skill_mastery_level(skill_id)
            if mastery < 0.6:
                skill = self.curriculum_tree.skills[skill_id]
                recommendations['skill_gaps'].append({
                    'skill_id': skill_id,
                    'name': skill.name,
                    'mastery_level': mastery,
                    'estimated_time': skill.estimated_time
                })

        return recommendations