
"""
Local file-based PersistenceService.

This replaces the MongoDB/Redis-backed implementation with a disk-backed
local storage under server-ml/local_storage/. Methods preserve the async
interface and method names so the rest of the codebase (controllers, app)
can remain unchanged.
"""

import asyncio
import hashlib
import json
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import glob
import torch

from models.persistence import LearningStateType
from utils.logging_config import get_logger, log_persistence_operation

logger = get_logger(__name__)


class PersistenceService:
    """Disk-backed persistence service that stores states and checkpoints in files."""

    def __init__(self, mongodb_uri: str = None, redis_uri: str = None, instance_id: Optional[str] = None):
        # Keep signature compatible but ignore DB URIs
        self.instance_id = instance_id or str(uuid.uuid4())
        # Base storage directory (server-ml/local_storage)
        self.base_dir = Path(__file__).resolve().parents[1] / "local_storage"
        self.learning_dir = self.base_dir / "learning_states"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.versions_dir = self.base_dir / "versions"
        self.exports_dir = self.base_dir / "exports"
        self.backups_dir = self.base_dir / "backups"
        self.locks_dir = self.base_dir / "locks"
        self.shared_dir = self.base_dir / "shared_states"

        self.is_coordinator = False
        self.version_counters: Dict[str, int] = {}

    async def initialize(self):
        """Create local storage directories and perform a quick health check."""
        try:
            for d in [self.learning_dir, self.checkpoints_dir, self.versions_dir,
                      self.exports_dir, self.backups_dir, self.locks_dir, self.shared_dir]:
                d.mkdir(parents=True, exist_ok=True)

            # create a file to store shared_state_version if not exists
            shared_version_file = self.base_dir / "shared_state_version.txt"
            if not shared_version_file.exists():
                shared_version_file.write_text("0")

            logger.info(f"Persistence service (local) initialized for instance {self.instance_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize local persistence service: {e}")
            raise

    def _now_iso(self) -> str:
        return datetime.utcnow().isoformat()

    async def _write_json(self, path: Path, obj: Any):
        await asyncio.to_thread(path.write_text, json.dumps(obj, default=str))

    async def _read_json(self, path: Path) -> Any:
        text = await asyncio.to_thread(path.read_text)
        return json.loads(text)

    async def save_learning_state(self, state_type: LearningStateType, data: Dict[str, Any],
                                  description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        start_time = time.time()
        try:
            log_persistence_operation("save", state_type.value, "started", {
                'description': description,
                'data_size': len(str(data))
            })

            version = await self._generate_version(state_type)
            checksum = self._calculate_checksum(data)

            state_folder = self.learning_dir / state_type.value
            state_folder.mkdir(parents=True, exist_ok=True)

            doc = {
                'state_type': state_type.value,
                'instance_id': self.instance_id,
                'version': version,
                'timestamp': self._now_iso(),
                'data': data,
                'metadata': metadata or {},
                'checksum': checksum
            }

            file_path = state_folder / f"{version}.json"
            await self._write_json(file_path, doc)

            # update version history
            await self._update_version_history(state_type, version, description)

            duration = time.time() - start_time
            log_persistence_operation("save", state_type.value, "success", {
                'version': version,
                'size_mb': len(str(data)) / (1024 * 1024),
                'duration': duration
            })

            logger.info(f"✅ Successfully saved {state_type.value} state | Version: {version} | Duration: {duration:.3f}s")
            return version

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Failed to save learning state: {e} | Duration: {duration:.3f}s")
            log_persistence_operation("save", state_type.value, "failed", {'error': str(e), 'duration': duration})
            raise

    async def load_learning_state(self, state_type: LearningStateType, version: Optional[str] = None,
                                  instance_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            folder = self.learning_dir / state_type.value
            if not folder.exists():
                return None

            if version:
                file_path = folder / f"{version}.json"
                if not file_path.exists():
                    return None
                doc = await self._read_json(file_path)
            else:
                # pick the latest by filename (versions encoded with timestamp_prefix)
                files = sorted(folder.glob("*.json"), reverse=True)
                if not files:
                    return None
                doc = await self._read_json(files[0])

            # verify checksum
            if doc.get('checksum') and not self._verify_checksum(doc.get('data'), doc.get('checksum')):
                logger.warning(f"Checksum verification failed for {state_type.value} version {doc.get('version')}")
                return None

            return {
                'data': doc.get('data'),
                'version': doc.get('version'),
                'timestamp': doc.get('timestamp'),
                'metadata': doc.get('metadata', {})
            }

        except Exception as e:
            logger.error(f"Failed to load learning state: {e}")
            raise

    async def save_model_checkpoint(self, model_name: str, model: torch.nn.Module,
                                    optimizer: Optional[torch.optim.Optimizer] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> str:
        try:
            version = await self._generate_version(LearningStateType.LLM_MODEL, model_name)
            model_folder = self.checkpoints_dir / model_name
            model_folder.mkdir(parents=True, exist_ok=True)

            ckpt_path = model_folder / f"{model_name}_{version}.pt"
            payload = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'metadata': metadata or {},
                'version': version,
                'timestamp': self._now_iso()
            }

            # torch.save may be blocking, run in thread
            await asyncio.to_thread(torch.save, payload, str(ckpt_path))

            logger.info(f"Saved model checkpoint {model_name} version {version}")
            return version

        except Exception as e:
            logger.error(f"Failed to save model checkpoint: {e}")
            raise

    async def load_model_checkpoint(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            model_folder = self.checkpoints_dir / model_name
            if not model_folder.exists():
                return None

            if version:
                ckpt_path = model_folder / f"{model_name}_{version}.pt"
                if not ckpt_path.exists():
                    return None
            else:
                files = sorted(model_folder.glob("*.pt"), reverse=True)
                if not files:
                    return None
                ckpt_path = files[0]

            payload = await asyncio.to_thread(torch.load, str(ckpt_path))

            return {
                'model_state_dict': payload.get('model_state_dict'),
                'optimizer_state_dict': payload.get('optimizer_state_dict'),
                'metadata': payload.get('metadata'),
                'version': payload.get('version'),
                'timestamp': payload.get('timestamp')
            }

        except Exception as e:
            logger.error(f"Failed to load model checkpoint: {e}")
            raise

    async def list_state_versions(self, state_type: LearningStateType, instance_id: Optional[str] = None,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        try:
            folder = self.learning_dir / state_type.value
            if not folder.exists():
                return []

            files = sorted(folder.glob("*.json"), reverse=True)[:limit]
            versions = []
            for f in files:
                doc = await self._read_json(f)
                versions.append({
                    'version': doc.get('version'),
                    'timestamp': doc.get('timestamp'),
                    'description': doc.get('metadata', {}).get('description'),
                    'performance_metrics': doc.get('metadata', {}).get('performance_metrics', {}),
                    'size_bytes': len(json.dumps(doc.get('data', {})))
                })

            return versions

        except Exception as e:
            logger.error(f"Failed to list state versions: {e}")
            raise

    async def export_learning_state(self, components: List[str] = None, format: str = "json",
                                    include_history: bool = True, compress: bool = False) -> Dict[str, Any]:
        try:
            if components is None:
                components = [s.value for s in LearningStateType]

            export_data = {
                'export_info': {
                    'instance_id': self.instance_id,
                    'timestamp': self._now_iso(),
                    'format': format,
                    'components': components,
                    'compressed': compress
                },
                'data': {}
            }

            for component in components:
                try:
                    state_type = LearningStateType(component)
                    state_data = await self.load_learning_state(state_type)
                    if state_data:
                        export_data['data'][component] = state_data
                except Exception as e:
                    logger.warning(f"Failed to export component {component}: {e}")

            export_id = f"export_{int(time.time())}"
            export_file = self.exports_dir / f"{export_id}.json"
            await self._write_json(export_file, export_data)

            return {
                'export_id': export_id,
                'data': export_data,
                'format': format,
                'size_bytes': len(json.dumps(export_data))
            }

        except Exception as e:
            logger.error(f"Failed to export learning state: {e}")
            raise

    async def import_learning_state(self, import_data: Dict[str, Any], components: List[str] = None,
                                    overwrite_existing: bool = False, validate_data: bool = True) -> Dict[str, Any]:
        try:
            if validate_data:
                # minimal validation
                if 'export_info' not in import_data or 'data' not in import_data:
                    return {'success': False, 'message': 'Import data missing export_info or data', 'validation_errors': ['missing export_info or data']}

            if components is None:
                components = import_data.get('export_info', {}).get('components', [])

            imported = []
            for component in components:
                if component in import_data.get('data', {}):
                    st = LearningStateType(component)
                    compdata = import_data['data'][component]['data'] if 'data' in import_data['data'][component] else import_data['data'][component]
                    await self.save_learning_state(st, compdata, description=f"Imported from {import_data.get('export_info', {}).get('instance_id', 'unknown')}")
                    imported.append(component)

            return {'success': True, 'imported_components': imported, 'message': f'Successfully imported {len(imported)} components'}

        except Exception as e:
            logger.error(f"Failed to import learning state: {e}")
            raise

    async def create_backup(self, backup_name: str, components: List[str] = None, compress: bool = True) -> Dict[str, Any]:
        try:
            export_result = await self.export_learning_state(components=components, format='json', include_history=True, compress=compress)
            checksum = self._calculate_checksum(export_result['data'])
            backup_id = f"backup_{int(time.time())}"
            backup_file = self.backups_dir / f"{backup_id}.json"
            backup_doc = {
                'backup_name': backup_name,
                'instance_id': self.instance_id,
                'components': export_result['data']['export_info']['components'],
                'data': export_result['data'],
                'compressed': compress,
                'size_bytes': export_result['size_bytes'],
                'checksum': checksum,
                'timestamp': self._now_iso()
            }
            await self._write_json(backup_file, backup_doc)
            return {'success': True, 'backup_id': backup_id, 'backup_name': backup_name, 'size_bytes': export_result['size_bytes'], 'message': f'Backup {backup_name} created successfully'}

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise

    async def restore_backup(self, backup_id: str, components: List[str] = None, confirm_restore: bool = False) -> Dict[str, Any]:
        try:
            if not confirm_restore:
                return {'success': False, 'message': 'Restore not confirmed. Set confirm_restore=true to proceed.'}

            backup_file = self.backups_dir / f"{backup_id}.json"
            if not backup_file.exists():
                raise ValueError(f"Backup {backup_id} not found")

            backup = await self._read_json(backup_file)
            if not self._verify_checksum(backup.get('data'), backup.get('checksum')):
                raise ValueError(f"Backup {backup_id} checksum verification failed")

            res = await self.import_learning_state(backup.get('data'), components=components, overwrite_existing=True, validate_data=False)
            return {'success': True, 'backup_id': backup_id, 'restored_components': res.get('imported_components', []), 'message': f"Successfully restored from backup {backup.get('backup_name', backup_id)}"}

        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            raise

    async def rollback_state(self, state_type: LearningStateType, target_version: str, confirm_rollback: bool = False) -> Dict[str, Any]:
        try:
            if not confirm_rollback:
                return {'success': False, 'message': 'Rollback not confirmed. Set confirm_rollback=true to proceed.'}

            target = await self.load_learning_state(state_type, target_version)
            if not target:
                raise ValueError(f"Target version {target_version} not found")

            current = await self.load_learning_state(state_type)
            current_version = current['version'] if current else None

            if current:
                await self.save_learning_state(state_type, current['data'], description=f"Auto-backup before rollback from {current_version} to {target_version}")

            new_version = await self.save_learning_state(state_type, target['data'], description=f"Rolled back to version {target_version}", metadata={'rollback': True, 'from_version': current_version, 'to_version': target_version})

            return {'success': True, 'state_type': state_type.value, 'rolled_back_from': current_version, 'rolled_back_to': target_version, 'new_version': new_version, 'message': f"Successfully rolled back to version {target_version}"}

        except Exception as e:
            logger.error(f"Failed to rollback state: {e}")
            raise

    async def sync_shared_state(self, force_sync: bool = False) -> Dict[str, Any]:
        """Basic local shared-state sync using files. Not distributed beyond local machine."""
        try:
            synced = []
            shared_version_file = self.base_dir / "shared_state_version.txt"
            current_version = int(shared_version_file.read_text()) if shared_version_file.exists() else 0

            # minimal sync: write this instance's last sync
            instance_file = self.base_dir / f"instance_{self.instance_id}.json"
            await self._write_json(instance_file, {'last_sync': time.time()})

            # bump version
            new_version = current_version + 1
            shared_version_file.write_text(str(new_version))

            return {'success': True, 'synced_components': synced, 'shared_state_version': new_version, 'conflicts_resolved': 0, 'message': 'Synchronized locally'}

        except Exception as e:
            logger.error(f"Failed to sync shared state: {e}")
            raise

    async def acquire_lock(self, lock_name: str, ttl_seconds: int = 30) -> bool:
        try:
            lock_file = self.locks_dir / f"{lock_name}.lock"
            now = datetime.utcnow()
            if lock_file.exists():
                data = json.loads(lock_file.read_text())
                expires = datetime.fromisoformat(data.get('expires_at'))
                if expires > now:
                    return False

            lock_doc = {'lock_name': lock_name, 'instance_id': self.instance_id, 'expires_at': (now + timedelta(seconds=ttl_seconds)).isoformat(), 'acquired_at': now.isoformat()}
            await self._write_json(lock_file, lock_doc)
            return True

        except Exception as e:
            logger.error(f"Failed to acquire lock {lock_name}: {e}")
            return False

    async def release_lock(self, lock_name: str) -> bool:
        try:
            lock_file = self.locks_dir / f"{lock_name}.lock"
            if not lock_file.exists():
                return False
            data = json.loads(lock_file.read_text())
            if data.get('instance_id') != self.instance_id:
                # not owner
                return False
            lock_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to release lock {lock_name}: {e}")
            return False

    async def get_distributed_status(self) -> Dict[str, Any]:
        try:
            instances = []
            for f in glob.glob(str(self.base_dir / 'instance_*.json')):
                try:
                    data = json.loads(Path(f).read_text())
                    instances.append(Path(f).stem.replace('instance_', ''))
                except Exception:
                    continue

            shared_version_file = self.base_dir / "shared_state_version.txt"
            shared_version = int(shared_version_file.read_text()) if shared_version_file.exists() else 0

            locks = [p.stem.replace('.lock', '') for p in self.locks_dir.glob('*.lock')]

            return {'instance_id': self.instance_id, 'is_coordinator': self.is_coordinator, 'active_instances': instances, 'shared_state_version': shared_version, 'last_sync': None, 'locks_held': locks}

        except Exception as e:
            logger.error(f"Failed to get distributed status: {e}")
            raise

    async def _generate_version(self, state_type: LearningStateType, suffix: str = "") -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        counter_key = f"{state_type.value}_{suffix}"
        if counter_key not in self.version_counters:
            self.version_counters[counter_key] = 0
        self.version_counters[counter_key] += 1
        counter = self.version_counters[counter_key]
        return f"{timestamp}_{counter}"

    async def _update_version_history(self, state_type: LearningStateType, version: str, description: Optional[str] = None):
        try:
            hist_file = self.versions_dir / f"{state_type.value}_history.json"
            history = {'instance_id': self.instance_id, 'state_type': state_type.value, 'versions': [], 'current_version': version}
            if hist_file.exists():
                history = json.loads(hist_file.read_text())

            version_info = {'version': version, 'timestamp': self._now_iso(), 'description': description}
            history.setdefault('versions', []).append(version_info)
            history['current_version'] = version
            # keep last 50
            if len(history['versions']) > 50:
                history['versions'] = history['versions'][-50:]
            await self._write_json(hist_file, history)
        except Exception as e:
            logger.warning(f"Failed to update version history: {e}")

    def _calculate_checksum(self, data: Any) -> str:
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _verify_checksum(self, data: Any, checksum: str) -> bool:
        return self._calculate_checksum(data) == checksum

    async def close(self):
        # nothing to close for local storage
        logger.info("Persistence service (local) closed")