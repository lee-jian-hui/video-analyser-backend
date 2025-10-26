"""
File Storage Service

Manages video file storage in OS-appropriate user directories.
For desktop app, stores files in user's Documents folder.
"""

import os
import uuid
import logging
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict

from storage_paths import get_videos_dir, get_storage_root

logger = logging.getLogger(__name__)


class FileStorage:
    """Manages video file storage in OS-appropriate directories"""

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize FileStorage.

        Args:
            base_dir: Optional custom base directory. If not provided,
                     uses OS-appropriate user directory.
        """
        storage_root = get_storage_root(base_dir)
        self.base_dir = get_videos_dir(storage_root)
        self.files: Dict[str, str] = {}  # file_id -> file_path (in-memory for now)

        logger.info(f"FileStorage initialized: {self.base_dir}")

    def save_uploaded_file(self, file_data: bytes, filename: str) -> Tuple[str, str]:
        """
        Save uploaded file and return (file_id, file_path).

        Args:
            file_data: Raw bytes of the video file
            filename: Original filename

        Returns:
            Tuple of (file_id, file_path)
        """
        file_id = uuid.uuid4().hex
        file_path = self.base_dir / f"{file_id}_{self._sanitize_filename(filename)}"

        # Save file
        try:
            with open(file_path, 'wb') as f:
                f.write(file_data)

            # Track in memory
            self.files[file_id] = str(file_path)

            logger.info(f"Saved file: {filename} → {file_id} ({len(file_data)} bytes)")

            return file_id, str(file_path)

        except Exception as e:
            logger.error(f"Failed to save file {filename}: {e}")
            raise

    def import_local_file(
        self,
        source_path: str,
        filename: Optional[str] = None,
        copy_file: bool = True
    ) -> Tuple[str, str]:
        """
        Register an existing local file by copying (or referencing) it.

        Args:
            source_path: Absolute path to the local file selected by the user.
            filename: Optional override for stored filename.
            copy_file: If True, copy into managed storage. Otherwise track original path.

        Returns:
            Tuple of (file_id, stored_file_path)
        """
        path = Path(source_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        display_name = filename or path.name
        file_id = uuid.uuid4().hex

        if copy_file:
            safe_name = self._sanitize_filename(display_name)
            suffix = Path(display_name).suffix or path.suffix or ""
            dest_path = self.base_dir / f"{file_id}_{safe_name}{suffix}"
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_path)
            tracked_path = dest_path
        else:
            tracked_path = path

        self.files[file_id] = str(tracked_path)
        logger.info(
            "Registered local file: %s → %s (copied=%s, size=%s bytes)",
            display_name,
            file_id,
            copy_file,
            path.stat().st_size,
        )

        return file_id, str(tracked_path)

    def get_file_path(self, file_id: str) -> str:
        """
        Get file path by file_id

        Args:
            file_id: Unique file identifier

        Returns:
            Absolute path to the file

        Raises:
            FileNotFoundError: If file_id not found
        """
        if file_id not in self.files:
            # Try to find file in directory (in case of restart)
            for file in self.base_dir.glob(f"{file_id}_*"):
                file_path = str(file)
                self.files[file_id] = file_path
                return file_path

            raise FileNotFoundError(f"File {file_id} not found")

        return self.files[file_id]

    def file_exists(self, file_id: str) -> bool:
        """Check if a file exists"""
        try:
            file_path = self.get_file_path(file_id)
            return Path(file_path).exists()
        except FileNotFoundError:
            return False

    def delete_file(self, file_id: str) -> bool:
        """
        Delete a video file.

        Args:
            file_id: Unique file identifier

        Returns:
            True if deleted, False if file not found
        """
        try:
            file_path = self.get_file_path(file_id)
            path = Path(file_path)

            if path.exists():
                path.unlink()
                logger.info(f"Deleted file: {file_id}")

            # Remove from tracking
            if file_id in self.files:
                del self.files[file_id]

            return True

        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {file_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            raise

    def forget_file(self, file_id: str) -> None:
        """Remove a file from in-memory tracking without touching disk."""
        if file_id in self.files:
            del self.files[file_id]

    def list_files(self) -> Dict[str, str]:
        """
        List all tracked files.

        Returns:
            Dict mapping file_id to file_path
        """
        return self.files.copy()

    def get_file_info(self, file_id: str) -> Dict[str, any]:
        """
        Get metadata about a file.

        Args:
            file_id: Unique file identifier

        Returns:
            Dict with file metadata (name, size, etc.)
        """
        file_path = self.get_file_path(file_id)
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File {file_id} does not exist on disk")

        return {
            "file_id": file_id,
            "filename": path.name,
            "file_path": str(path),
            "size_bytes": path.stat().st_size,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "created_at": path.stat().st_ctime,
            "modified_at": path.stat().st_mtime
        }

    def cleanup_old_files(self, days: int = 7):
        """
        Delete files older than specified days.

        Args:
            days: Delete files older than this many days
        """
        import time
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)

        deleted_count = 0

        for file in self.base_dir.glob("*_*"):
            if file.stat().st_mtime < cutoff_time:
                try:
                    file_id = file.name.split('_')[0]
                    self.delete_file(file_id)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to cleanup file {file.name}: {e}")

        logger.info(f"Cleaned up {deleted_count} old files (older than {days} days)")
        return deleted_count

    def _sanitize_filename(self, filename: str) -> str:
        """Strip unsafe characters from filenames to keep storage predictable."""
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
        return safe_filename or "video"
