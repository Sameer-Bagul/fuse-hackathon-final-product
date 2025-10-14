"""
Persistence controller for handling persistence-related API endpoints.
Provides REST API for saving/loading learning states, model checkpoints,
and distributed learning coordination.
"""

from typing import List, Optional, Dict, Any
from fastapi import HTTPException, BackgroundTasks
from datetime import datetime
import gzip
import json
import hashlib
from collections import defaultdict, deque

from models.persistence import (
    LearningStateType, SaveStateRequest, SaveStateResponse, LoadStateRequest,
    LoadStateResponse, ListVersionsRequest, ListVersionsResponse, VersionInfo,
    ExportStateRequest, ExportStateResponse, ImportStateRequest, ImportStateResponse,
    RollbackRequest, RollbackResponse, BackupRequest, BackupResponse,
    RestoreRequest, RestoreResponse, DistributedStatusResponse, SyncRequest, SyncResponse
)
from services.persistence_service import PersistenceService
from models.learner import Memory  # Import Memory class for reset
from utils.logging_config import get_logger

logger = get_logger(__name__)


class PersistenceController:
    """Controller for persistence operations"""

    def __init__(self, persistence_service: PersistenceService,
                  component_registry: Optional[Dict[str, Any]] = None):
        self.persistence_service = persistence_service
        self.component_registry = component_registry or {}
        self.rollback_points: Dict[LearningStateType, Dict[str, Any]] = {}

    async def save_learning_state(self, request: SaveStateRequest) -> SaveStateResponse:
        """Save current learning state"""
        try:
            logger.info(f"[Persistence API] Saving {request.state_type.value} state")

            # Collect data based on state type
            data = await self._collect_state_data(request.state_type, request.include_related)

            # Save the state
            version = await self.persistence_service.save_learning_state(
                state_type=request.state_type,
                data=data,
                description=request.description
            )

            logger.info(f"[Persistence API] Successfully saved {request.state_type.value} state version {version}")

            return SaveStateResponse(
                success=True,
                version=version,
                state_type=request.state_type,
                timestamp=datetime.utcnow(),
                message=f"Successfully saved {request.state_type.value} state"
            )

        except Exception as e:
            logger.error(f"[Persistence API] Failed to save learning state: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save learning state: {str(e)}")

    async def load_learning_state(self, request: LoadStateRequest) -> LoadStateResponse:
        """Load learning state from database"""
        try:
            logger.info(f"[Persistence API] Loading {request.state_type.value} state")

            # Load the state
            state_data = await self.persistence_service.load_learning_state(
                state_type=request.state_type,
                version=request.version,
                instance_id=request.instance_id
            )

            if not state_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"State {request.state_type.value} version {request.version or 'latest'} not found"
                )

            # Apply loaded data to current components
            await self._apply_state_data(request.state_type, state_data["data"])

            logger.info(f"[Persistence API] Successfully loaded {request.state_type.value} state version {state_data['version']}")

            return LoadStateResponse(
                success=True,
                state_type=request.state_type,
                version=state_data["version"],
                timestamp=state_data["timestamp"],
                data=state_data["data"],
                message=f"Successfully loaded {request.state_type.value} state"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[Persistence API] Failed to load learning state: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load learning state: {str(e)}")

    async def list_state_versions(self, request: ListVersionsRequest) -> ListVersionsResponse:
        """List available versions for a state type"""
        try:
            logger.info(f"[Persistence API] Listing versions for {request.state_type.value}")

            versions_data = await self.persistence_service.list_state_versions(
                state_type=request.state_type,
                instance_id=request.instance_id,
                limit=request.limit
            )

            # Convert to response format
            versions = []
            current_version = None

            for v_data in versions_data:
                version_info = VersionInfo(
                    version=v_data["version"],
                    timestamp=v_data["timestamp"],
                    description=v_data.get("description"),
                    performance_metrics=v_data.get("performance_metrics", {}),
                    size_bytes=v_data.get("size_bytes")
                )
                versions.append(version_info)

                if not current_version:
                    current_version = v_data["version"]

            logger.info(f"[Persistence API] Found {len(versions)} versions for {request.state_type.value}")

            return ListVersionsResponse(
                state_type=request.state_type,
                versions=versions,
                total_count=len(versions),
                current_version=current_version
            )

        except Exception as e:
            logger.error(f"[Persistence API] Failed to list state versions: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list state versions: {str(e)}")

    async def export_learning_state(self, request: ExportStateRequest) -> ExportStateResponse:
        """Export learning state"""
        try:
            logger.info(f"[Persistence API] Exporting learning state with components: {request.components}")

            export_result = await self.persistence_service.export_learning_state(
                components=request.components,
                format=request.format,
                include_history=request.include_history,
                compress=request.compress
            )

            logger.info(f"[Persistence API] Successfully exported learning state, size: {export_result['size_bytes']} bytes")

            return ExportStateResponse(
                success=True,
                export_id=export_result["export_id"],
                format=request.format,
                components=export_result["data"]["export_info"]["components"],
                size_bytes=export_result["size_bytes"],
                data=export_result["data"] if request.format == "json" else None,
                message=f"Successfully exported {len(export_result['data']['export_info']['components'])} components"
            )

        except Exception as e:
            logger.error(f"[Persistence API] Failed to export learning state: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to export learning state: {str(e)}")

    async def import_learning_state(self, request: ImportStateRequest) -> ImportStateResponse:
        """Import learning state"""
        try:
            logger.info(f"[Persistence API] Importing learning state with {len(request.components)} components")

            import_result = await self.persistence_service.import_learning_state(
                import_data=request.data,
                components=request.components,
                overwrite_existing=request.overwrite_existing,
                validate_data=request.validate_data
            )

            if not import_result["success"]:
                return ImportStateResponse(
                    success=False,
                    imported_components=[],
                    version="",
                    message=import_result["message"],
                    validation_errors=import_result.get("validation_errors", [])
                )

            logger.info(f"[Persistence API] Successfully imported {len(import_result['imported_components'])} components")

            return ImportStateResponse(
                success=True,
                imported_components=import_result["imported_components"],
                version="",  # Will be set by individual component saves
                message=import_result["message"]
            )

        except Exception as e:
            logger.error(f"[Persistence API] Failed to import learning state: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to import learning state: {str(e)}")

    async def rollback_state(self, request: RollbackRequest) -> RollbackResponse:
        """Rollback to a previous version"""
        try:
            logger.info(f"[Persistence API] Rolling back {request.state_type.value} to version {request.target_version}")

            rollback_result = await self.persistence_service.rollback_state(
                state_type=request.state_type,
                target_version=request.target_version,
                confirm_rollback=request.confirm_rollback
            )

            if not rollback_result["success"]:
                return RollbackResponse(
                    success=False,
                    state_type=request.state_type,
                    rolled_back_from="",
                    rolled_back_to=request.target_version,
                    timestamp=datetime.utcnow(),
                    message=rollback_result["message"]
                )

            logger.info(f"[Persistence API] Successfully rolled back {request.state_type.value}")

            return RollbackResponse(
                success=True,
                state_type=request.state_type,
                rolled_back_from=rollback_result["rolled_back_from"],
                rolled_back_to=rollback_result["rolled_back_to"],
                timestamp=datetime.utcnow(),
                message=rollback_result["message"]
            )

        except Exception as e:
            logger.error(f"[Persistence API] Failed to rollback state: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to rollback state: {str(e)}")

    async def create_backup(self, request: BackupRequest) -> BackupResponse:
        """Create a backup"""
        try:
            logger.info(f"[Persistence API] Creating backup '{request.backup_name}'")

            backup_result = await self.persistence_service.create_backup(
                backup_name=request.backup_name,
                components=request.components,
                compress=request.compress
            )

            logger.info(f"[Persistence API] Successfully created backup '{request.backup_name}'")

            return BackupResponse(
                success=True,
                backup_id=backup_result["backup_id"],
                backup_name=request.backup_name,
                components=backup_result["components"],
                size_bytes=backup_result["size_bytes"],
                timestamp=datetime.utcnow(),
                message=backup_result["message"]
            )

        except Exception as e:
            logger.error(f"[Persistence API] Failed to create backup: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create backup: {str(e)}")

    async def restore_backup(self, request: RestoreRequest) -> RestoreResponse:
        """Restore from backup"""
        try:
            logger.info(f"[Persistence API] Restoring from backup {request.backup_id}")

            restore_result = await self.persistence_service.restore_backup(
                backup_id=request.backup_id,
                components=request.components,
                confirm_restore=request.confirm_restore
            )

            if not restore_result["success"]:
                return RestoreResponse(
                    success=False,
                    backup_id=request.backup_id,
                    restored_components=[],
                    timestamp=datetime.utcnow(),
                    message=restore_result["message"]
                )

            logger.info(f"[Persistence API] Successfully restored from backup {request.backup_id}")

            return RestoreResponse(
                success=True,
                backup_id=request.backup_id,
                restored_components=restore_result["restored_components"],
                timestamp=datetime.utcnow(),
                message=restore_result["message"]
            )

        except Exception as e:
            logger.error(f"[Persistence API] Failed to restore backup: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to restore backup: {str(e)}")

    async def get_distributed_status(self) -> DistributedStatusResponse:
        """Get distributed learning status"""
        try:
            logger.info("[Persistence API] Getting distributed status")

            status = await self.persistence_service.get_distributed_status()

            return DistributedStatusResponse(**status)

        except Exception as e:
            logger.error(f"[Persistence API] Failed to get distributed status: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get distributed status: {str(e)}")

    async def sync_shared_state(self, request: SyncRequest) -> SyncResponse:
        """Manually trigger shared state synchronization"""
        try:
            logger.info("[Persistence API] Triggering manual sync")

            sync_result = await self.persistence_service.sync_shared_state(
                force_sync=request.force_sync
            )

            logger.info(f"[Persistence API] Sync completed, synced {len(sync_result['synced_components'])} components")

            return SyncResponse(
                success=True,
                synced_components=sync_result["synced_components"],
                shared_state_version=sync_result["shared_state_version"],
                conflicts_resolved=sync_result["conflicts_resolved"],
                timestamp=datetime.utcnow(),
                message=sync_result["message"]
            )

        except Exception as e:
            logger.error(f"[Persistence API] Failed to sync shared state: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to sync shared state: {str(e)}")

    async def reset_all_data(self, confirm_reset: bool = False) -> Dict[str, Any]:
        """Reset all learning data, history, analytics, and state"""
        try:
            if not confirm_reset:
                raise HTTPException(
                    status_code=400,
                    detail="Reset not confirmed. Set confirm_reset=true to proceed with full data reset."
                )

            logger.info("[Persistence API] Starting full data reset")

            reset_results = {}

            # Reset LLM state
            try:
                await self._reset_llm_state()
                reset_results["llm_model"] = "reset successfully"
                logger.info("LLM state reset successfully")
            except Exception as e:
                reset_results["llm_model"] = f"failed: {str(e)}"
                logger.error(f"Failed to reset LLM state: {e}")

            # Reset meta-learning state
            try:
                await self._reset_meta_learning_state()
                reset_results["meta_learning"] = "reset successfully"
                logger.info("Meta-learning state reset successfully")
            except Exception as e:
                reset_results["meta_learning"] = f"failed: {str(e)}"
                logger.error(f"Failed to reset meta-learning state: {e}")

            # Reset history
            try:
                await self._reset_history_state()
                reset_results["history"] = "reset successfully"
                logger.info("History reset successfully")
            except Exception as e:
                reset_results["history"] = f"failed: {str(e)}"
                logger.error(f"Failed to reset history: {e}")

            # Reset curriculum
            try:
                await self._reset_curriculum_state()
                reset_results["curriculum"] = "reset successfully"
                logger.info("Curriculum state reset successfully")
            except Exception as e:
                reset_results["curriculum"] = f"failed: {str(e)}"
                logger.error(f"Failed to reset curriculum: {e}")

            # Reset reward system
            try:
                await self._reset_reward_state()
                reset_results["reward_system"] = "reset successfully"
                logger.info("Reward system reset successfully")
            except Exception as e:
                reset_results["reward_system"] = f"failed: {str(e)}"
                logger.error(f"Failed to reset reward system: {e}")

            # Reset analytics
            try:
                await self._reset_analytics_state()
                reset_results["analytics"] = "reset successfully"
                logger.info("Analytics reset successfully")
            except Exception as e:
                reset_results["analytics"] = f"failed: {str(e)}"
                logger.error(f"Failed to reset analytics: {e}")

            # Reset feedback
            try:
                await self._reset_feedback_state()
                reset_results["feedback"] = "reset successfully"
                logger.info("Feedback reset successfully")
            except Exception as e:
                reset_results["feedback"] = f"failed: {str(e)}"
                logger.error(f"Failed to reset feedback: {e}")

            # Clear persistence storage - only if persistence service has the method
            try:
                if hasattr(self.persistence_service, 'clear_all_data'):
                    await self.persistence_service.clear_all_data()
                    reset_results["persistence_storage"] = "cleared successfully"
                    logger.info("Persistence storage cleared successfully")
                else:
                    reset_results["persistence_storage"] = "not available (method missing)"
                    logger.warning("Persistence service does not have clear_all_data method")
            except Exception as e:
                reset_results["persistence_storage"] = f"failed: {str(e)}"
                logger.error(f"Failed to clear persistence storage: {e}")

            success_count = sum(1 for result in reset_results.values() if "successfully" in str(result))
            total_count = len(reset_results)

            logger.info(f"[Persistence API] Full data reset completed: {success_count}/{total_count} components reset successfully")

            return {
                "success": success_count == total_count,
                "message": f"Reset completed: {success_count}/{total_count} components reset successfully",
                "reset_results": reset_results,
                "timestamp": datetime.utcnow().isoformat()
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[Persistence API] Failed to reset all data: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to reset all data: {str(e)}")

    async def _reset_llm_state(self):
        """Reset LLM to initial state"""
        llm = self.component_registry.get('llm')
        if not llm:
            raise ValueError("LLM component not available in registry")

        # Reset PPO agent to initial state
        if hasattr(llm, 'ppo_agent'):
            # Reset policy network parameters using the PPO agent's reset method
            if hasattr(llm.ppo_agent, 'clear_buffer'):
                llm.ppo_agent.clear_buffer()
            # Reset curriculum state
            if hasattr(llm.ppo_agent, 'curriculum_state'):
                llm.ppo_agent.curriculum_state = {
                    'current_skill': None,
                    'difficulty_level': 0,
                    'task_progress': 0,
                    'success_streak': 0
                }

        # Clear memory and patterns
        if hasattr(llm, 'memory'):
            if hasattr(llm.memory, 'clear_memory'):
                llm.memory.clear_memory()
            else:
                # Fallback for different memory structure
                llm.memory = Memory()  # Create new memory instance
        if hasattr(llm, 'patterns'):
            llm.patterns.clear()
        if hasattr(llm, 'episode_rewards'):
            llm.episode_rewards.clear()

        # Reset dynamic parameters to defaults
        if hasattr(llm, 'dynamic_parameters'):
            llm.dynamic_parameters.clear()

        # Reset learning context
        if hasattr(llm, 'learning_context'):
            llm.learning_context.clear()

    async def _reset_meta_learning_state(self):
        """Reset meta-learning to initial state"""
        meta_learning_service = self.component_registry.get('meta_learning_service')
        if not meta_learning_service:
            raise ValueError("Meta-learning service not available in registry")

        meta_learner = meta_learning_service.meta_learner

        # Reset to default strategy
        from models.meta_learning import LearningStrategy
        meta_learner.current_strategy = LearningStrategy.ADAPTIVE

        # Reset parameters to defaults
        meta_learner.current_params = {
            'exploration_rate': 0.1,
            'learning_rate': 0.001,
            'adaptation_threshold': 0.05
        }

        # Reset metrics
        from models.meta_learning import MetaMetrics
        meta_learner.metrics = MetaMetrics()

        # Clear adaptation rules and history
        meta_learner.adaptation_rules.clear()
        meta_learner.strategy_history.clear()

    async def _reset_history_state(self):
        """Reset history to initial state"""
        history = self.component_registry.get('history')
        if not history:
            raise ValueError("History component not available in registry")

        # Clear all interactions
        history.interactions.clear()

    async def _reset_curriculum_state(self):
        """Reset curriculum to initial state"""
        scheduler = self.component_registry.get('scheduler')
        if not scheduler:
            raise ValueError("Scheduler component not available in registry")

        # Clear learner progress
        scheduler.learner_progress.clear()

        # Reset curriculum tree if it exists
        if hasattr(scheduler, 'curriculum_tree') and scheduler.curriculum_tree:
            # Reinitialize curriculum tree (this depends on the curriculum implementation)
            pass  # Curriculum tree reset would depend on specific implementation

    async def _reset_reward_state(self):
        """Reset reward system to initial state"""
        reward_service = self.component_registry.get('reward_service')
        if not reward_service:
            raise ValueError("Reward service not available in registry")

        # Reset to default weights
        default_weights = {
            'accuracy': 0.4,
            'coherence': 0.3,
            'factuality': 0.2,
            'creativity': 0.1
        }
        reward_service.configure_weights(default_weights)

        # Clear reward history
        if hasattr(reward_service, 'reward_history'):
            reward_service.reward_history.clear()

    async def _reset_analytics_state(self):
        """Reset analytics to initial state"""
        analytics_service = self.component_registry.get('analytics_service')
        if not analytics_service:
            raise ValueError("Analytics service not available in registry")

        # Clear analytics data (implementation depends on analytics service)
        if hasattr(analytics_service, 'clear_data'):
            analytics_service.clear_data()

    async def _reset_feedback_state(self):
        """Reset feedback to initial state"""
        feedback_service = self.component_registry.get('feedback_service')
        if not feedback_service:
            raise ValueError("Feedback service not available in registry")

        # Clear feedback history and preferences
        if hasattr(feedback_service, 'clear_all_feedback'):
            feedback_service.clear_all_feedback()

    async def _collect_state_data(self, state_type: LearningStateType, include_related: bool = True) -> Dict[str, Any]:
        """Collect current state data for a given state type with validation, metadata, and compression"""
        logger.info(f"Collecting state data for {state_type.value}")

        # Collect raw data
        if state_type == LearningStateType.LLM_MODEL:
            data = await self._collect_llm_state()
        elif state_type == LearningStateType.META_LEARNING:
            data = await self._collect_meta_learning_state()
        elif state_type == LearningStateType.HISTORY:
            data = await self._collect_history_state()
        elif state_type == LearningStateType.CURRICULUM:
            data = await self._collect_curriculum_state()
        elif state_type == LearningStateType.REWARD_SYSTEM:
            data = await self._collect_reward_state()
        elif state_type == LearningStateType.ANALYTICS:
            data = await self._collect_analytics_state()
        elif state_type == LearningStateType.FEEDBACK:
            data = await self._collect_feedback_state()
        elif state_type == LearningStateType.COMPLETE_SYSTEM:
            data = await self._collect_complete_system_state()
        else:
            raise ValueError(f"Unknown state type: {state_type}")

        # Validate data
        self._validate_state_data(state_type, data)

        # Create metadata
        metadata = self._create_metadata(data, state_type)

        # Compress data
        compressed_data = self._compress_data(data)

        logger.info(f"Successfully collected and compressed state data for {state_type.value}, size: {len(compressed_data)} bytes")

        return {
            "metadata": metadata,
            "compressed_data": compressed_data
        }

    async def _apply_state_data(self, state_type: LearningStateType, state_data: Dict[str, Any]):
        """Apply loaded state data to current components with rollback capabilities"""
        logger.info(f"Applying state data for {state_type.value}")

        # Extract metadata and compressed data
        metadata = state_data.get("metadata", {})
        compressed_data = state_data.get("compressed_data")

        if not compressed_data:
            raise ValueError("No compressed data found in state data")

        # Decompress data
        data = self._decompress_data(compressed_data)

        # Validate data
        self._validate_state_data(state_type, data)

        # Create rollback point
        rollback_point = await self._create_rollback_point(state_type)

        try:
            # Apply data
            if state_type == LearningStateType.LLM_MODEL:
                await self._apply_llm_state(data)
            elif state_type == LearningStateType.META_LEARNING:
                await self._apply_meta_learning_state(data)
            elif state_type == LearningStateType.HISTORY:
                await self._apply_history_state(data)
            elif state_type == LearningStateType.CURRICULUM:
                await self._apply_curriculum_state(data)
            elif state_type == LearningStateType.REWARD_SYSTEM:
                await self._apply_reward_state(data)
            elif state_type == LearningStateType.ANALYTICS:
                await self._apply_analytics_state(data)
            elif state_type == LearningStateType.FEEDBACK:
                await self._apply_feedback_state(data)
            elif state_type == LearningStateType.COMPLETE_SYSTEM:
                await self._apply_complete_system_state(data)

            logger.info(f"Successfully applied state data for {state_type.value}")

        except Exception as e:
            logger.error(f"Failed to apply state data for {state_type.value}, initiating rollback: {e}")
            await self._rollback_to_point(state_type, rollback_point)
            raise HTTPException(status_code=500, detail=f"Failed to apply state data and rolled back: {str(e)}")

    # Placeholder methods for state collection/application
    # These will need to be implemented with actual component access

    async def _collect_llm_state(self) -> Dict[str, Any]:
        """Collect LLM model state"""
        llm = self.component_registry.get('llm')
        if not llm:
            raise ValueError("LLM component not available in registry")

        return {
            "ppo_agent_state": {
                "policy_state_dict": self.persistence_service._serialize_model_state(
                    llm.ppo_agent.policy.state_dict()
                ),
                "optimizer_state_dict": (
                    self.persistence_service._serialize_optimizer_state(
                        llm.ppo_agent.optimizer.state_dict()
                    )
                    if hasattr(llm.ppo_agent, 'optimizer') else None
                ),
            },
            "memory": len(llm.memory.rewards) if hasattr(llm, 'memory') else 0,
            "patterns": dict(llm.patterns),
            "episode_rewards": list(llm.episode_rewards),
            "dynamic_parameters": llm.dynamic_parameters.copy(),
            "learning_context": llm.learning_context.copy()
        }

    async def _collect_meta_learning_state(self) -> Dict[str, Any]:
        """Collect meta-learning state"""
        meta_learning_service = self.component_registry.get('meta_learning_service')
        if not meta_learning_service:
            raise ValueError("Meta-learning service not available in registry")

        return {
            "current_strategy": meta_learning_service.meta_learner.current_strategy.value,
            "current_params": meta_learning_service.meta_learner.current_params.copy(),
            "metrics": meta_learning_service.meta_learner.metrics.to_dict(),
            "adaptation_rules": [rule.__dict__ for rule in meta_learning_service.meta_learner.adaptation_rules],
            "strategy_history": list(meta_learning_service.meta_learner.strategy_history)
        }

    async def _collect_history_state(self) -> Dict[str, Any]:
        """Collect history state"""
        history = self.component_registry.get('history')
        if not history:
            raise ValueError("History component not available in registry")

        return {
            "interactions": history.interactions.copy()
        }

    async def _collect_curriculum_state(self) -> Dict[str, Any]:
        """Collect curriculum state"""
        scheduler = self.component_registry.get('scheduler')
        if not scheduler:
            raise ValueError("Scheduler component not available in registry")

        return {
            "learner_progress": dict(scheduler.learner_progress),
            "curriculum_tree": scheduler.curriculum_tree.to_dict() if hasattr(scheduler, 'curriculum_tree') else {}
        }

    async def _collect_reward_state(self) -> Dict[str, Any]:
        """Collect reward system state"""
        reward_service = self.component_registry.get('reward_service')
        if not reward_service:
            raise ValueError("Reward service not available in registry")

        return {
            "weights": reward_service.get_current_weights(),
            "history": reward_service.get_history_summary()
        }

    async def _collect_analytics_state(self) -> Dict[str, Any]:
        """Collect analytics state"""
        analytics_service = self.component_registry.get('analytics_service')
        if not analytics_service:
            raise ValueError("Analytics service not available in registry")

        return {
            "metrics": analytics_service.get_current_metrics(),
            "insights": analytics_service.get_current_insights()
        }

    async def _collect_feedback_state(self) -> Dict[str, Any]:
        """Collect feedback state"""
        feedback_service = self.component_registry.get('feedback_service')
        if not feedback_service:
            raise ValueError("Feedback service not available in registry")

        return {
            "feedback_history": feedback_service.get_feedback_summary(),
            "preferences": feedback_service.get_learned_preferences()
        }

    async def _collect_complete_system_state(self) -> Dict[str, Any]:
        """Collect complete system state"""
        # Collect all components
        state = {}
        for state_type in LearningStateType:
            try:
                state[state_type.value] = await self._collect_state_data(state_type, False)
            except Exception as e:
                logger.warning(f"Failed to collect {state_type.value} state: {e}")
        return state

    async def _apply_llm_state(self, data: Dict[str, Any]):
        """Apply LLM state"""
        llm = self.component_registry.get('llm')
        if not llm:
            raise ValueError("LLM component not available in registry")

        if "ppo_agent_state" in data:
            agent_state = data["ppo_agent_state"]
            if "policy_state_dict" in agent_state:
                policy_state = self.persistence_service._deserialize_model_state(agent_state["policy_state_dict"])
                llm.ppo_agent.policy.load_state_dict(policy_state)
            if "optimizer_state_dict" in agent_state and agent_state["optimizer_state_dict"]:
                optimizer_state = self.persistence_service._deserialize_optimizer_state(agent_state["optimizer_state_dict"])
                llm.ppo_agent.optimizer.load_state_dict(optimizer_state)

        # Restore other attributes
        if "patterns" in data:
            llm.patterns = defaultdict(int, data["patterns"])
        if "episode_rewards" in data:
            llm.episode_rewards = deque(data["episode_rewards"], maxlen=100)
        if "dynamic_parameters" in data:
            llm.dynamic_parameters.update(data["dynamic_parameters"])
        if "learning_context" in data:
            llm.learning_context.update(data["learning_context"])

    async def _apply_meta_learning_state(self, data: Dict[str, Any]):
        """Apply meta-learning state"""
        meta_learning_service = self.component_registry.get('meta_learning_service')
        if not meta_learning_service:
            raise ValueError("Meta-learning service not available in registry")

        meta_learner = meta_learning_service.meta_learner

        if "current_strategy" in data:
            from models.meta_learning import LearningStrategy
            meta_learner.current_strategy = LearningStrategy(data["current_strategy"])
        if "current_params" in data:
            meta_learner.current_params.update(data["current_params"])
        if "metrics" in data:
            from models.meta_learning import MetaMetrics
            meta_learner.metrics = MetaMetrics.from_dict(data["metrics"])

    async def _apply_history_state(self, data: Dict[str, Any]):
        """Apply history state"""
        history = self.component_registry.get('history')
        if not history:
            raise ValueError("History component not available in registry")

        if "interactions" in data:
            history.interactions = data["interactions"]

    async def _apply_curriculum_state(self, data: Dict[str, Any]):
        """Apply curriculum state"""
        scheduler = self.component_registry.get('scheduler')
        if not scheduler:
            raise ValueError("Scheduler component not available in registry")

        if "learner_progress" in data:
            scheduler.learner_progress = data["learner_progress"]

    async def _apply_reward_state(self, data: Dict[str, Any]):
        """Apply reward state"""
        reward_service = self.component_registry.get('reward_service')
        if not reward_service:
            raise ValueError("Reward service not available in registry")

        if "weights" in data:
            reward_service.configure_weights(data["weights"])

    async def _apply_analytics_state(self, data: Dict[str, Any]):
        """Apply analytics state"""
        # Analytics state is typically read-only, just log
        logger.info("Analytics state loaded (read-only)")

    async def _apply_feedback_state(self, data: Dict[str, Any]):
        """Apply feedback state"""
        feedback_service = self.component_registry.get('feedback_service')
        if not feedback_service:
            raise ValueError("Feedback service not available in registry")

        # Feedback state restoration depends on the feedback service implementation
        logger.info("Feedback state loaded")

    async def _apply_complete_system_state(self, data: Dict[str, Any]):
        """Apply complete system state"""
        for state_type_str, state_data in data.items():
            try:
                state_type = LearningStateType(state_type_str)
                await self._apply_state_data(state_type, state_data)
            except Exception as e:
                logger.warning(f"Failed to apply {state_type_str} state: {e}")
    async def _validate_state_data(self, state_type: LearningStateType, data: Dict[str, Any]):
        """Validate state data structure and required fields"""
        if not isinstance(data, dict):
            raise ValueError(f"State data for {state_type.value} must be a dictionary")

        if state_type == LearningStateType.LLM_MODEL:
            required_keys = ["ppo_agent_state", "patterns", "episode_rewards", "dynamic_parameters", "learning_context"]
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key '{key}' in LLM state data")
            if "ppo_agent_state" in data and not isinstance(data["ppo_agent_state"], dict):
                raise ValueError("ppo_agent_state must be a dictionary")

        elif state_type == LearningStateType.META_LEARNING:
            required_keys = ["current_strategy", "current_params", "metrics", "adaptation_rules", "strategy_history"]
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key '{key}' in meta-learning state data")

        elif state_type == LearningStateType.HISTORY:
            if "interactions" not in data:
                raise ValueError("Missing required key 'interactions' in history state data")

        elif state_type == LearningStateType.CURRICULUM:
            if "learner_progress" not in data:
                raise ValueError("Missing required key 'learner_progress' in curriculum state data")

        elif state_type == LearningStateType.REWARD_SYSTEM:
            if "weights" not in data:
                raise ValueError("Missing required key 'weights' in reward system state data")

        elif state_type == LearningStateType.ANALYTICS:
            if "metrics" not in data:
                raise ValueError("Missing required key 'metrics' in analytics state data")

        elif state_type == LearningStateType.FEEDBACK:
            if "feedback_history" not in data:
                raise ValueError("Missing required key 'feedback_history' in feedback state data")

        elif state_type == LearningStateType.COMPLETE_SYSTEM:
            # For complete system, validate that it's a dict of state types
            for key, value in data.items():
                try:
                    sub_state_type = LearningStateType(key)
                    self._validate_state_data(sub_state_type, value)
                except ValueError as e:
                    raise ValueError(f"Invalid sub-state {key} in complete system: {e}")

        logger.debug(f"Validation passed for {state_type.value} state data")

    def _create_metadata(self, data: Dict[str, Any], state_type: LearningStateType) -> Dict[str, Any]:
        """Create metadata for state data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        size_bytes = len(data_str.encode())

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "state_type": state_type.value,
            "version": "1.0",
            "data_hash": data_hash,
            "size_bytes": size_bytes,
            "compression": "gzip"
        }

    def _compress_data(self, data: Dict[str, Any]) -> bytes:
        """Compress state data using gzip"""
        data_str = json.dumps(data, default=str)
        return gzip.compress(data_str.encode())

    def _decompress_data(self, compressed_data: bytes) -> Dict[str, Any]:
        """Decompress state data"""
        data_str = gzip.decompress(compressed_data).decode()
        return json.loads(data_str)

    async def _create_rollback_point(self, state_type: LearningStateType) -> Dict[str, Any]:
        """Create a rollback point by collecting current state"""
        logger.debug(f"Creating rollback point for {state_type.value}")

        if state_type == LearningStateType.LLM_MODEL:
            rollback_data = await self._collect_llm_state()
        elif state_type == LearningStateType.META_LEARNING:
            rollback_data = await self._collect_meta_learning_state()
        elif state_type == LearningStateType.HISTORY:
            rollback_data = await self._collect_history_state()
        elif state_type == LearningStateType.CURRICULUM:
            rollback_data = await self._collect_curriculum_state()
        elif state_type == LearningStateType.REWARD_SYSTEM:
            rollback_data = await self._collect_reward_state()
        elif state_type == LearningStateType.ANALYTICS:
            rollback_data = await self._collect_analytics_state()
        elif state_type == LearningStateType.FEEDBACK:
            rollback_data = await self._collect_feedback_state()
        elif state_type == LearningStateType.COMPLETE_SYSTEM:
            rollback_data = await self._collect_complete_system_state()
        else:
            raise ValueError(f"Unknown state type for rollback: {state_type}")

        self.rollback_points[state_type] = rollback_data
        return rollback_data

    async def _rollback_to_point(self, state_type: LearningStateType, rollback_data: Dict[str, Any]):
        """Rollback to a previous state point"""
        logger.warning(f"Rolling back {state_type.value} to previous state")

        try:
            if state_type == LearningStateType.LLM_MODEL:
                await self._apply_llm_state(rollback_data)
            elif state_type == LearningStateType.META_LEARNING:
                await self._apply_meta_learning_state(rollback_data)
            elif state_type == LearningStateType.HISTORY:
                await self._apply_history_state(rollback_data)
            elif state_type == LearningStateType.CURRICULUM:
                await self._apply_curriculum_state(rollback_data)
            elif state_type == LearningStateType.REWARD_SYSTEM:
                await self._apply_reward_state(rollback_data)
            elif state_type == LearningStateType.ANALYTICS:
                await self._apply_analytics_state(rollback_data)
            elif state_type == LearningStateType.FEEDBACK:
                await self._apply_feedback_state(rollback_data)
            elif state_type == LearningStateType.COMPLETE_SYSTEM:
                await self._apply_complete_system_state(rollback_data)

            logger.info(f"Successfully rolled back {state_type.value}")

        except Exception as e:
            logger.error(f"Failed to rollback {state_type.value}: {e}")
            raise
