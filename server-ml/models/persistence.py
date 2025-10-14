"""
Database models and schemas for persistence layer.
Handles MongoDB document structures for learning state persistence.
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import json


class LearningStateType(str, Enum):
    """Types of learning states that can be persisted"""
    LLM_MODEL = "llm_model"
    META_LEARNING = "meta_learning"
    HISTORY = "history"
    CURRICULUM = "curriculum"
    REWARD_SYSTEM = "reward_system"
    ANALYTICS = "analytics"
    FEEDBACK = "feedback"
    COMPLETE_SYSTEM = "complete_system"


class CheckpointMetadata(BaseModel):
    """Metadata for model checkpoints"""
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    training_episodes: int = 0
    model_architecture: Dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class LearningStateDocument(BaseModel):
    """Base document structure for learning state persistence"""
    id: Optional[str] = Field(alias="_id", default=None)
    state_type: LearningStateType
    instance_id: str  # Identifier for the learning instance
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    checksum: Optional[str] = None  # For data integrity verification


class ModelCheckpointDocument(BaseModel):
    """Document structure for PyTorch model checkpoints"""
    id: Optional[str] = Field(alias="_id", default=None)
    model_name: str
    instance_id: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_state_dict: Dict[str, Any]  # Serialized PyTorch state dict
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    metadata: CheckpointMetadata
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class VersionHistoryDocument(BaseModel):
    """Document for tracking version history"""
    id: Optional[str] = Field(alias="_id", default=None)
    instance_id: str
    state_type: LearningStateType
    versions: List[Dict[str, Any]] = Field(default_factory=list)  # List of version info
    current_version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DistributedLockDocument(BaseModel):
    """Document for distributed locking mechanism"""
    id: Optional[str] = Field(alias="_id", default=None)
    lock_name: str
    instance_id: str
    acquired_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    lock_data: Dict[str, Any] = Field(default_factory=dict)


class SharedStateDocument(BaseModel):
    """Document for shared state across distributed instances"""
    id: Optional[str] = Field(alias="_id", default=None)
    state_key: str
    instance_id: str
    data: Dict[str, Any]
    version: int = 1
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    checksum: Optional[str] = None


class ExportImportDocument(BaseModel):
    """Document structure for export/import operations"""
    id: Optional[str] = Field(alias="_id", default=None)
    operation_type: str  # "export" or "import"
    instance_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    format: str  # "json", "pickle", "binary"
    components: List[str]  # List of component names included
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"  # "pending", "completed", "failed"


class BackupDocument(BaseModel):
    """Document structure for backup operations"""
    id: Optional[str] = Field(alias="_id", default=None)
    backup_name: str
    instance_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: List[str]
    data: Dict[str, Any]
    compressed: bool = False
    size_bytes: int = 0
    checksum: str
    status: str = "active"  # "active", "archived", "deleted"


# Request/Response models for API endpoints
class SaveStateRequest(BaseModel):
    """Request model for saving learning state"""
    state_type: LearningStateType
    description: Optional[str] = None
    include_related: bool = True  # Include related components


class SaveStateResponse(BaseModel):
    """Response model for saving learning state"""
    success: bool
    version: str
    state_type: LearningStateType
    timestamp: datetime
    message: str


class LoadStateRequest(BaseModel):
    """Request model for loading learning state"""
    state_type: LearningStateType
    version: Optional[str] = None  # If None, load latest
    instance_id: Optional[str] = None


class LoadStateResponse(BaseModel):
    """Response model for loading learning state"""
    success: bool
    state_type: LearningStateType
    version: str
    timestamp: datetime
    data: Dict[str, Any]
    message: str


class ListVersionsRequest(BaseModel):
    """Request model for listing state versions"""
    state_type: LearningStateType
    instance_id: Optional[str] = None
    limit: int = 10


class VersionInfo(BaseModel):
    """Information about a specific version"""
    version: str
    timestamp: datetime
    description: Optional[str] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    size_bytes: Optional[int] = None


class ListVersionsResponse(BaseModel):
    """Response model for listing state versions"""
    state_type: LearningStateType
    versions: List[VersionInfo]
    total_count: int
    current_version: Optional[str] = None


class ExportStateRequest(BaseModel):
    """Request model for exporting learning state"""
    components: List[str] = Field(default_factory=list)  # Empty means all components
    format: str = "json"
    include_history: bool = True
    compress: bool = False


class ExportStateResponse(BaseModel):
    """Response model for exporting learning state"""
    success: bool
    export_id: str
    format: str
    components: List[str]
    size_bytes: int
    download_url: Optional[str] = None
    data: Optional[Dict[str, Any]] = None  # For direct data return


class ImportStateRequest(BaseModel):
    """Request model for importing learning state"""
    data: Dict[str, Any]
    format: str = "json"
    components: List[str] = Field(default_factory=list)
    overwrite_existing: bool = False
    validate_data: bool = True


class ImportStateResponse(BaseModel):
    """Response model for importing learning state"""
    success: bool
    imported_components: List[str]
    version: str
    message: str
    validation_errors: List[str] = Field(default_factory=list)


class RollbackRequest(BaseModel):
    """Request model for rolling back to a previous version"""
    state_type: LearningStateType
    target_version: str
    confirm_rollback: bool = False


class RollbackResponse(BaseModel):
    """Response model for rollback operations"""
    success: bool
    state_type: LearningStateType
    rolled_back_from: str
    rolled_back_to: str
    timestamp: datetime
    message: str


class BackupRequest(BaseModel):
    """Request model for creating backups"""
    backup_name: str
    components: List[str] = Field(default_factory=list)
    compress: bool = True


class BackupResponse(BaseModel):
    """Response model for backup operations"""
    success: bool
    backup_id: str
    backup_name: str
    components: List[str]
    size_bytes: int
    timestamp: datetime
    message: str


class RestoreRequest(BaseModel):
    """Request model for restoring from backup"""
    backup_id: str
    components: List[str] = Field(default_factory=list)
    confirm_restore: bool = False


class RestoreResponse(BaseModel):
    """Response model for restore operations"""
    success: bool
    backup_id: str
    restored_components: List[str]
    timestamp: datetime
    message: str


class DistributedStatusResponse(BaseModel):
    """Response model for distributed learning status"""
    instance_id: str
    is_coordinator: bool
    active_instances: List[str]
    shared_state_version: int
    last_sync: Optional[datetime] = None
    locks_held: List[str] = Field(default_factory=list)


class SyncRequest(BaseModel):
    """Request model for manual synchronization"""
    force_sync: bool = False
    components: List[str] = Field(default_factory=list)


class SyncResponse(BaseModel):
    """Response model for synchronization operations"""
    success: bool
    synced_components: List[str]
    shared_state_version: int
    conflicts_resolved: int = 0
    timestamp: datetime
    message: str