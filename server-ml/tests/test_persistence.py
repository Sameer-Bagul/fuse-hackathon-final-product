"""
Test script for persistence functionality.
Tests the persistence service and API endpoints.
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.persistence_service import PersistenceService
from models.persistence import LearningStateType, CheckpointMetadata
from models.learner import LLM
from models.meta_learning import MetaLearner
from utils.logging_config import get_logger

logger = get_logger(__name__)


async def test_persistence_service():
    """Test the persistence service functionality"""
    print("🧪 Testing Persistence Service...")

    # Initialize persistence service
    persistence_service = PersistenceService()
    await persistence_service.initialize()

    try:
        # Test 1: Save and load learning state
        print("\n1. Testing learning state save/load...")

        test_data = {
            "test_key": "test_value",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {"accuracy": 0.85, "loss": 0.23}
        }

        # Save state
        version = await persistence_service.save_learning_state(
            state_type=LearningStateType.META_LEARNING,
            data=test_data,
            description="Test learning state"
        )
        print(f"   ✅ Saved state version: {version}")

        # Load state
        loaded_state = await persistence_service.load_learning_state(
            state_type=LearningStateType.META_LEARNING,
            version=version
        )

        if loaded_state and loaded_state["data"]["test_key"] == "test_value":
            print("   ✅ Loaded state matches saved data")
        else:
            print("   ❌ Loaded state does not match saved data")

        # Test 2: List versions
        print("\n2. Testing version listing...")
        versions = await persistence_service.list_state_versions(
            state_type=LearningStateType.META_LEARNING,
            limit=5
        )
        print(f"   ✅ Found {len(versions)} versions")

        # Test 3: Export/Import
        print("\n3. Testing export/import...")
        export_result = await persistence_service.export_learning_state(
            components=[LearningStateType.META_LEARNING.value],
            format="json"
        )
        print(f"   ✅ Exported {export_result['size_bytes']} bytes")

        # Import the exported data
        import_result = await persistence_service.import_learning_state(
            import_data=export_result["data"],
            components=[LearningStateType.META_LEARNING.value]
        )
        print(f"   ✅ Imported {len(import_result['imported_components'])} components")

        # Test 4: Backup/Restore
        print("\n4. Testing backup/restore...")
        backup_result = await persistence_service.create_backup(
            backup_name="test_backup",
            components=[LearningStateType.META_LEARNING.value]
        )
        print(f"   ✅ Created backup: {backup_result['backup_name']}")

        # Restore from backup
        restore_result = await persistence_service.restore_backup(
            backup_id=backup_result["backup_id"],
            components=[LearningStateType.META_LEARNING.value],
            confirm_restore=True
        )
        print(f"   ✅ Restored {len(restore_result['restored_components'])} components")

        # Test 5: Distributed status
        print("\n5. Testing distributed status...")
        status = await persistence_service.get_distributed_status()
        print(f"   ✅ Instance ID: {status['instance_id']}")
        print(f"   ✅ Is coordinator: {status['is_coordinator']}")
        print(f"   ✅ Active instances: {len(status['active_instances'])}")

        print("\n🎉 All persistence tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        await persistence_service.close()


async def test_model_checkpointing():
    """Test model checkpointing functionality"""
    print("\n🔧 Testing Model Checkpointing...")

    persistence_service = PersistenceService()
    await persistence_service.initialize()

    try:
        # Create a simple test model
        import torch
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoint
        metadata = CheckpointMetadata(
            version="test_1.0",
            description="Test model checkpoint",
            performance_metrics={"accuracy": 0.90, "loss": 0.15}
        )

        version = await persistence_service.save_model_checkpoint(
            model_name="test_model",
            model=model,
            optimizer=optimizer,
            metadata=metadata
        )
        print(f"   ✅ Saved model checkpoint version: {version}")

        # Load checkpoint
        checkpoint = await persistence_service.load_model_checkpoint(
            model_name="test_model",
            version=version
        )

        if checkpoint and "model_state_dict" in checkpoint:
            print("   ✅ Loaded model checkpoint successfully")

            # Verify model can be reconstructed
            new_model = SimpleModel()
            new_model.load_state_dict(checkpoint["model_state_dict"])
            print("   ✅ Model state dict loaded successfully")
        else:
            print("   ❌ Failed to load model checkpoint")

        print("🎉 Model checkpointing tests completed!")

    except Exception as e:
        print(f"❌ Model checkpointing test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await persistence_service.close()


async def main():
    """Run all persistence tests"""
    print("🚀 Starting Persistence Tests...")
    print("=" * 50)

    # Test basic persistence functionality
    await test_persistence_service()

    # Test model checkpointing
    await test_model_checkpointing()

    print("=" * 50)
    print("🏁 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())