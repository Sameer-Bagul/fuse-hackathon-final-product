import sys
import os
import logging
from typing import List, Optional, Dict, Any, Tuple
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.task_generator import TaskGenerator
from models.prompt import Prompt
from models.curriculum import CurriculumProgress, CurriculumTree, DifficultyLevel

logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, task_generator=None, curriculum_tree=None, feedback_service=None):
        try:
            self.task_generator = task_generator or TaskGenerator()
            self.curriculum_tree = curriculum_tree or CurriculumTree.create_default_curriculum()
            self.feedback_service = feedback_service  # For feedback-based curriculum adjustments
            self.learner_progress = {}  # learner_id -> CurriculumProgress
            self.task_history = []  # Track scheduled tasks for analysis
            logger.info("Scheduler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Scheduler: {str(e)}")
            raise

    def set_curriculum_tree(self, curriculum_tree: CurriculumTree):
        """Set the curriculum tree for scheduling decisions"""
        self.curriculum_tree = curriculum_tree
        self.task_generator.set_curriculum_tree(curriculum_tree)

    def register_learner(self, learner_id: str) -> CurriculumProgress:
        """Register a new learner and create their progress tracking"""
        if learner_id not in self.learner_progress:
            self.learner_progress[learner_id] = CurriculumProgress(learner_id)
            logger.info(f"Registered new learner: {learner_id}")
        return self.learner_progress[learner_id]

    def get_learner_progress(self, learner_id: str) -> Optional[CurriculumProgress]:
        """Get progress for a specific learner"""
        return self.learner_progress.get(learner_id)

    def update_learner_progress(self, learner_id: str, skill_id: str, score: float, time_spent: int = 0):
        """Update learner progress after task completion"""
        if learner_id not in self.learner_progress:
            self.register_learner(learner_id)

        progress = self.learner_progress[learner_id]
        progress.update_skill_progress(skill_id, score, time_spent)

        # Trigger curriculum adjustments
        self._adjust_curriculum_for_learner(learner_id)

        logger.info(f"Updated progress for learner {learner_id}: skill {skill_id}, score {score:.3f}")

    def schedule(self, num_tasks: int, prompt=None, learner_id: Optional[str] = None,
                target_difficulty: Optional[DifficultyLevel] = None):
        """
        Schedule curriculum-aware tasks for a learner

        Args:
            num_tasks: Number of tasks to schedule
            prompt: Optional prompt for context
            learner_id: ID of the learner for personalization
            target_difficulty: Override difficulty level
        """
        try:
            if not isinstance(num_tasks, int) or num_tasks <= 0:
                raise ValueError("num_tasks must be a positive integer")

            if num_tasks > 50:
                raise ValueError("num_tasks cannot exceed 50 for performance reasons")

            logger.info(f"Scheduling {num_tasks} tasks for learner {learner_id or 'anonymous'}")

            # Get or create learner progress
            learner_progress = None
            if learner_id:
                learner_progress = self.get_learner_progress(learner_id) or self.register_learner(learner_id)

            tasks = []
            task_metadata = []

            for i in range(num_tasks):
                try:
                    # Generate curriculum-aware task
                    task = self.task_generator.generate_task(
                        prompt=prompt,
                        learner_progress=learner_progress,
                        target_difficulty=target_difficulty
                    )

                    if task is None:
                        logger.warning(f"Task generation returned None for task {i}")
                        continue

                    tasks.append(task)

                    # Extract metadata from task generator
                    if hasattr(self.task_generator, 'generated_tasks') and self.task_generator.generated_tasks:
                        latest_task = self.task_generator.generated_tasks[-1]
                        if isinstance(latest_task, dict):
                            task_metadata.append(latest_task)
                        else:
                            task_metadata.append({'task': task, 'index': i})
                    else:
                        task_metadata.append({'task': task, 'index': i})

                except Exception as e:
                    logger.warning(f"Failed to generate task {i}: {str(e)}")
                    continue

            if not tasks:
                raise RuntimeError("Failed to generate any valid tasks")

            # Prioritize tasks with curriculum awareness
            try:
                prioritized_tasks, prioritized_metadata = self._prioritize_curriculum_tasks(
                    tasks, task_metadata, learner_progress
                )

                # Store scheduling history
                self.task_history.append({
                    'learner_id': learner_id,
                    'num_tasks': len(prioritized_tasks),
                    'timestamp': os.times()[4] if hasattr(os, 'times') else 0,
                    'metadata': prioritized_metadata
                })

                logger.info(f"Successfully scheduled {len(prioritized_tasks)} curriculum-aware tasks")
                return prioritized_tasks

            except Exception as e:
                logger.error(f"Failed to prioritize tasks: {str(e)}")
                return tasks

        except ValueError as e:
            logger.warning(f"Validation error in schedule: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in schedule: {str(e)}")
            raise RuntimeError(f"Task scheduling failed: {str(e)}")

    def _prioritize_curriculum_tasks(self, tasks: List[List[float]],
                                   task_metadata: List[Dict[str, Any]],
                                   learner_progress: Optional[CurriculumProgress]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """Prioritize tasks based on curriculum requirements and learner progress"""

        if not learner_progress:
            # Fallback to basic prioritization
            prioritized_tasks = self.task_generator.prioritize_tasks(tasks)
            return prioritized_tasks, task_metadata

        # Enhanced prioritization with curriculum awareness
        task_scores = []

        for i, (task, metadata) in enumerate(zip(tasks, task_metadata)):
            base_score = sum(task) / len(task)

            # Curriculum-based scoring
            curriculum_score = self._calculate_curriculum_priority(metadata, learner_progress)
            skill_gap_score = self._calculate_skill_gap_priority(metadata, learner_progress)
            prerequisite_score = self._calculate_prerequisite_priority(metadata, learner_progress)
            progress_score = self._calculate_progress_alignment_score(metadata, learner_progress)

            # Combine scores (weighted)
            total_score = (
                base_score * 0.3 +
                curriculum_score * 0.3 +
                skill_gap_score * 0.2 +
                prerequisite_score * 0.1 +
                progress_score * 0.1
            )

            task_scores.append((total_score, i))

        # Sort by total score descending
        task_scores.sort(reverse=True)

        # Reorder tasks and metadata
        prioritized_tasks = [tasks[i] for _, i in task_scores]
        prioritized_metadata = [task_metadata[i] for _, i in task_scores]

        return prioritized_tasks, prioritized_metadata

    def _calculate_curriculum_priority(self, metadata: Dict[str, Any],
                                     learner_progress: CurriculumProgress) -> float:
        """Calculate priority based on curriculum position"""
        skill_id = metadata.get('target_skill')
        if not skill_id or skill_id not in self.curriculum_tree.skills:
            return 0.5

        skill = self.curriculum_tree.skills[skill_id]

        # Higher priority for skills at learner's current difficulty level
        learner_difficulty = learner_progress.get_recommended_difficulty()
        if skill.difficulty == learner_difficulty:
            return 1.0
        elif abs(list(DifficultyLevel).index(skill.difficulty) - list(DifficultyLevel).index(learner_difficulty)) == 1:
            return 0.7  # Adjacent difficulty levels
        else:
            return 0.3  # Too easy or too hard

    def _calculate_skill_gap_priority(self, metadata: Dict[str, Any],
                                    learner_progress: CurriculumProgress) -> float:
        """Calculate priority based on skill gaps that need filling"""
        skill_id = metadata.get('target_skill')
        if not skill_id:
            return 0.5

        mastery_level = learner_progress.get_skill_mastery_level(skill_id)

        # Higher priority for skills with lower mastery
        if mastery_level < 0.4:
            return 1.0  # High priority for struggling skills
        elif mastery_level < 0.6:
            return 0.8  # Medium priority
        elif mastery_level < 0.8:
            return 0.5  # Low priority
        else:
            return 0.2  # Already mastered

    def _calculate_prerequisite_priority(self, metadata: Dict[str, Any],
                                       learner_progress: CurriculumProgress) -> float:
        """Calculate priority based on prerequisite relationships"""
        skill_id = metadata.get('target_skill')
        if not skill_id or skill_id not in self.curriculum_tree.skills:
            return 0.5

        skill = self.curriculum_tree.skills[skill_id]

        # Check if prerequisites are met
        prereqs_met = all(learner_progress.is_skill_mastered(prereq) for prereq in skill.prerequisites)

        if not prereqs_met:
            return 0.3  # Lower priority if prerequisites not met
        else:
            return 0.8  # Higher priority if ready to learn

    def _calculate_progress_alignment_score(self, metadata: Dict[str, Any],
                                          learner_progress: CurriculumProgress) -> float:
        """Calculate how well the task aligns with learner's progress trajectory"""
        skill_id = metadata.get('target_skill')
        if not skill_id:
            return 0.5

        # Get next recommended skills
        next_skills = self.curriculum_tree.get_next_skills(learner_progress.completed_skills)

        if skill_id in next_skills[:3]:  # Top 3 recommended skills
            return 1.0
        elif skill_id in next_skills:
            return 0.7
        else:
            return 0.3

    def _adjust_curriculum_for_learner(self, learner_id: str):
        """Adjust curriculum parameters based on learner performance and feedback"""
        progress = self.learner_progress.get(learner_id)
        if not progress:
            return

        # Get recent task history for this learner
        recent_tasks = [h for h in self.task_history if h['learner_id'] == learner_id][-10:]

        # Get user preferences and feedback insights
        feedback_adjustments = {}
        if self.feedback_service:
            try:
                preferences = self.feedback_service.get_user_preferences(
                    type('Request', (), {'user_id': learner_id, 'include_history': False})()
                ).preferences

                # Adjust curriculum based on user preferences
                feedback_adjustments = self._apply_feedback_based_curriculum_adjustments(
                    learner_id, progress, preferences
                )

                logger.info(f"Applied feedback-based curriculum adjustments for learner {learner_id}")
            except Exception as e:
                logger.warning(f"Failed to get feedback for curriculum adjustment: {str(e)}")

        if recent_tasks:
            # Extract recent performance (this would need to be enhanced with actual performance data)
            # For now, use a simple heuristic
            recent_performance = [0.5 + (i * 0.05) for i in range(len(recent_tasks))]  # Mock data

            # Adjust task generator curriculum with feedback insights
            self.task_generator.adjust_curriculum_based_on_performance(
                progress, recent_performance
            )

    def get_curriculum_status(self, learner_id: str) -> Dict[str, Any]:
        """Get comprehensive curriculum status for a learner"""
        progress = self.get_learner_progress(learner_id)
        if not progress:
            return {"error": "Learner not found"}

        available_skills = progress.get_available_skills(self.curriculum_tree)
        recommendations = self.task_generator.get_curriculum_recommendations(progress)

        return {
            "learner_id": learner_id,
            "progress": progress.to_dict(),
            "available_skills": available_skills,
            "recommendations": recommendations,
            "curriculum_summary": {
                "total_skills": len(self.curriculum_tree.skills),
                "completed_skills": len(progress.completed_skills),
                "completion_percentage": len(progress.completed_skills) / len(self.curriculum_tree.skills) * 100,
                "current_difficulty": progress.current_difficulty.value,
                "recommended_difficulty": progress.get_recommended_difficulty().value
            }
        }

    def validate_prerequisites(self, learner_id: str, skill_id: str) -> Dict[str, Any]:
        """Check if a learner meets prerequisites for a skill"""
        progress = self.get_learner_progress(learner_id)
        if not progress:
            return {"valid": False, "reason": "Learner not found"}

        if skill_id not in self.curriculum_tree.skills:
            return {"valid": False, "reason": "Skill not found in curriculum"}

        skill = self.curriculum_tree.skills[skill_id]
        unmet_prereqs = []

        for prereq_id in skill.prerequisites:
            if not progress.is_skill_mastered(prereq_id):
                prereq_skill = self.curriculum_tree.skills.get(prereq_id)
                unmet_prereqs.append({
                    "skill_id": prereq_id,
                    "name": prereq_skill.name if prereq_skill else prereq_id,
                    "mastery_level": progress.get_skill_mastery_level(prereq_id)
                })

        return {
            "valid": len(unmet_prereqs) == 0,
            "skill_id": skill_id,
            "skill_name": skill.name,
            "unmet_prerequisites": unmet_prereqs,
            "estimated_time_to_complete": skill.estimated_time
        }

    def _apply_feedback_based_curriculum_adjustments(self, learner_id: str,
                                                progress: CurriculumProgress,
                                                preferences) -> Dict[str, Any]:
        """Apply curriculum adjustments based on user feedback and preferences"""
        adjustments = {}

        try:
            # Adjust skill focus based on user preferences
            if hasattr(preferences, 'preference_weights') and preferences.preference_weights:
                # Map feedback categories to curriculum skills
                category_skill_mapping = {
                    'accuracy': ['data_analysis', 'algorithm_design'],
                    'coherence': ['logical_reasoning', 'communication'],
                    'factuality': ['research_methods', 'verification'],
                    'creativity': ['problem_solving', 'innovation'],
                    'usefulness': ['practical_application', 'implementation'],
                    'relevance': ['domain_knowledge', 'context_awareness'],
                    'completeness': ['comprehensive_analysis', 'detail_orientation'],
                    'clarity': ['communication', 'presentation']
                }

                skill_focus_adjustments = {}
                for category, weight in preferences.preference_weights.items():
                    if category in category_skill_mapping:
                        for skill_id in category_skill_mapping[category]:
                            skill_focus_adjustments[skill_id] = skill_focus_adjustments.get(skill_id, 0) + weight

                if skill_focus_adjustments:
                    adjustments['skill_focus'] = skill_focus_adjustments
                    logger.info(f"Applied skill focus adjustments for learner {learner_id}: {skill_focus_adjustments}")

            # Adjust difficulty based on user feedback patterns
            if self.feedback_service:
                try:
                    # Get recent feedback to assess difficulty preference
                    feedback_history = self.feedback_service.get_feedback_history(
                        type('Request', (), {
                            'user_id': learner_id,
                            'limit': 20,
                            'offset': 0,
                            'category_filter': None,
                            'feedback_type_filter': None
                        })()
                    )

                    if feedback_history.feedbacks:
                        # Analyze feedback patterns for difficulty adjustment
                        difficulty_adjustment = self._analyze_feedback_for_difficulty(
                            feedback_history.feedbacks
                        )
                        if difficulty_adjustment:
                            adjustments['difficulty_adjustment'] = difficulty_adjustment
                            logger.info(f"Applied difficulty adjustment for learner {learner_id}: {difficulty_adjustment}")

                except Exception as e:
                    logger.warning(f"Failed to analyze feedback for difficulty: {str(e)}")

            # Adjust pacing based on user engagement
            if hasattr(preferences, 'feedback_history') and preferences.feedback_history:
                engagement_rate = len(preferences.feedback_history) / max(1, (progress.total_time_spent // 3600))  # feedbacks per hour
                if engagement_rate > 2.0:  # High engagement
                    adjustments['pace_adjustment'] = 'faster'
                elif engagement_rate < 0.5:  # Low engagement
                    adjustments['pace_adjustment'] = 'slower'

        except Exception as e:
            logger.error(f"Error applying feedback-based curriculum adjustments: {str(e)}")

        return adjustments

    def _analyze_feedback_for_difficulty(self, feedbacks) -> Optional[str]:
        """Analyze feedback patterns to determine difficulty adjustments"""
        try:
            # Count different types of feedback
            rating_counts = {'too_easy': 0, 'appropriate': 0, 'too_hard': 0}

            for feedback in feedbacks:
                if hasattr(feedback, 'rating') and feedback.rating:
                    if feedback.rating >= 4:
                        rating_counts['too_easy'] += 1
                    elif feedback.rating <= 2:
                        rating_counts['too_hard'] += 1
                    else:
                        rating_counts['appropriate'] += 1

                # Also check comments for difficulty indicators
                if hasattr(feedback, 'comment') and feedback.comment:
                    comment_lower = feedback.comment.lower()
                    if any(word in comment_lower for word in ['easy', 'simple', 'basic', 'straightforward']):
                        rating_counts['too_easy'] += 0.5
                    elif any(word in comment_lower for word in ['hard', 'difficult', 'complex', 'challenging']):
                        rating_counts['too_hard'] += 0.5
                    elif any(word in comment_lower for word in ['appropriate', 'good', 'right']):
                        rating_counts['appropriate'] += 0.5

            # Determine adjustment based on counts
            total_signals = sum(rating_counts.values())
            if total_signals >= 3:  # Need minimum signals for reliable adjustment
                if rating_counts['too_easy'] > rating_counts['appropriate'] + rating_counts['too_hard']:
                    return 'increase_difficulty'
                elif rating_counts['too_hard'] > rating_counts['appropriate'] + rating_counts['too_easy']:
                    return 'decrease_difficulty'

        except Exception as e:
            logger.warning(f"Error analyzing feedback for difficulty: {str(e)}")

        return None