"""Short-lived, user-scoped file artifacts downloaded from Drive."""

import os
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any


class ArtifactError(ValueError):
    """Raised when an artifact is unavailable or belongs to another actor."""


@dataclass(frozen=True)
class DownloadArtifact:
    artifact_id: str
    user_id: str
    path: str
    metadata: dict[str, Any]
    created_at: float


class ArtifactStore:
    def __init__(self, ttl_seconds: int = 15 * 60, clock=time.monotonic):
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._artifacts: dict[str, DownloadArtifact] = {}
        self._lock = Lock()

    def register(self, user_id: str, download: dict[str, Any]) -> DownloadArtifact:
        path = download["temp_path"]
        if not os.path.isfile(path):
            raise ArtifactError("Downloaded temporary file is missing")
        self.cleanup_expired()
        artifact = DownloadArtifact(
            artifact_id=str(uuid.uuid4()),
            user_id=user_id,
            path=path,
            metadata={key: value for key, value in download.items() if key != "temp_path"},
            created_at=self.clock(),
        )
        with self._lock:
            self._artifacts[artifact.artifact_id] = artifact
        return artifact

    def consume(self, artifact_id: str, user_id: str) -> DownloadArtifact:
        self.cleanup_expired()
        with self._lock:
            artifact = self._artifacts.get(artifact_id)
            if artifact is None:
                raise ArtifactError("Download artifact was not found or has expired")
            if artifact.user_id != user_id:
                raise ArtifactError("Download artifact does not belong to this user")
            self._artifacts.pop(artifact_id)
        return artifact

    def cleanup_expired(self) -> None:
        now = self.clock()
        with self._lock:
            expired = [
                artifact_id
                for artifact_id, artifact in self._artifacts.items()
                if now - artifact.created_at >= self.ttl_seconds
            ]
            artifacts = [self._artifacts.pop(artifact_id) for artifact_id in expired]
        for artifact in artifacts:
            self.delete_file(artifact.path)

    @staticmethod
    def delete_file(path: str) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


_artifact_store = ArtifactStore()


def get_artifact_store() -> ArtifactStore:
    return _artifact_store
