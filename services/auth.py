"""JWT authentication and SQLite-backed users."""

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import jwt
from pwdlib import PasswordHash

from config import APP_DB_PATH, JWT_EXPIRE_MINUTES, JWT_SECRET

ROLE_SCOPES = {
    "admin": ["drive:read", "memory:read", "memory:write"],
    "user": ["drive:read", "memory:read"],
}


class AuthenticationError(PermissionError):
    """Raised when credentials or a JWT cannot be authenticated."""


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    username: str
    role: str
    scopes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuthService:
    def __init__(
        self,
        db_path: str = APP_DB_PATH,
        jwt_secret: str = JWT_SECRET,
        expire_minutes: int = JWT_EXPIRE_MINUTES,
    ):
        self.db_path = db_path
        self.jwt_secret = jwt_secret
        self.expire_minutes = expire_minutes
        self.password_hash = PasswordHash.recommended()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )
                """
            )

    def create_user(self, user_id: str, username: str, password: str, role: str) -> None:
        if role not in ROLE_SCOPES:
            raise ValueError(f"Unsupported role: {role}")
        if not user_id.strip() or not username.strip():
            raise ValueError("user_id and username are required")
        if len(password) < 8:
            raise ValueError("Password must contain at least 8 characters")

        password_hash = self.password_hash.hash(password)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (user_id, username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    user_id = excluded.user_id,
                    password_hash = excluded.password_hash,
                    role = excluded.role,
                    is_active = 1
                """,
                (user_id, username, password_hash, role, datetime.now(timezone.utc).isoformat()),
            )

    def authenticate(self, username: str, password: str) -> AuthenticatedUser:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,),
            ).fetchone()

        if not row or not row["is_active"] or not self.password_hash.verify(
            password, row["password_hash"]
        ):
            raise AuthenticationError("Invalid username or password")
        return self._user_from_row(row)

    def issue_token(self, user: AuthenticatedUser) -> tuple[str, int]:
        self._require_secret()
        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=self.expire_minutes)
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role,
            "scopes": user.scopes,
            "iat": now,
            "exp": expires,
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256"), self.expire_minutes * 60

    def verify_token(self, token: str) -> AuthenticatedUser:
        if not token:
            raise AuthenticationError("Authentication token is required")
        self._require_secret()
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("Authentication token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthenticationError("Authentication token is invalid") from exc

        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationError("Authentication token has no subject")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = ? AND is_active = 1",
                (user_id,),
            ).fetchone()
        if not row:
            raise AuthenticationError("User is inactive or no longer exists")
        return self._user_from_row(row)

    def _require_secret(self) -> None:
        if len(self.jwt_secret) < 32:
            raise RuntimeError("JWT_SECRET must contain at least 32 characters")

    @staticmethod
    def _user_from_row(row: sqlite3.Row) -> AuthenticatedUser:
        role = row["role"]
        return AuthenticatedUser(
            user_id=row["user_id"],
            username=row["username"],
            role=role,
            scopes=list(ROLE_SCOPES.get(role, [])),
        )


_auth_service: AuthService | None = None
_auth_lock = Lock()


def get_auth_service() -> AuthService:
    global _auth_service
    if _auth_service is None:
        with _auth_lock:
            if _auth_service is None:
                _auth_service = AuthService()
    return _auth_service
