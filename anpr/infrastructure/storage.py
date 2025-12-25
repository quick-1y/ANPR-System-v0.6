#/anpr/infrastructure/storage.py
#!/usr/bin/env python3
import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Sequence

import aiosqlite

from .logging_manager import get_logger


class EventDatabase:
    """SQLite-хранилище для последних распознанных номеров."""

    def __init__(self, db_path: str = "data/db/anpr.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()
        self.logger = get_logger(__name__)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection) -> None:
        """Добавляет отсутствующие столбцы без уничтожения существующих данных."""

        def _column_exists(name: str) -> bool:
            cursor = conn.execute("PRAGMA table_info(events)")
            return any(row[1] == name for row in cursor.fetchall())

        if not _column_exists("frame_path"):
            conn.execute("ALTER TABLE events ADD COLUMN frame_path TEXT")
        if not _column_exists("plate_path"):
            conn.execute("ALTER TABLE events ADD COLUMN plate_path TEXT")
        if not _column_exists("country"):
            conn.execute("ALTER TABLE events ADD COLUMN country TEXT")
        if not _column_exists("direction"):
            conn.execute("ALTER TABLE events ADD COLUMN direction TEXT")

    @staticmethod
    def _ensure_indexes(conn: sqlite3.Connection) -> None:
        """Гарантирует наличие индексов для ускорения выборок."""

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_channel ON events(channel)"
        )

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    plate TEXT NOT NULL,
                    country TEXT,
                    confidence REAL,
                    source TEXT,
                    frame_path TEXT,
                    plate_path TEXT,
                    direction TEXT
                )
                """
            )
            self._ensure_columns(conn)
            self._ensure_indexes(conn)
            conn.commit()

    def insert_event(
        self,
        channel: str,
        plate: str,
        country: Optional[str] = None,
        confidence: float = 0.0,
        source: str = "",
        timestamp: Optional[str] = None,
        frame_path: Optional[str] = None,
        plate_path: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> int:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                (
                    "INSERT INTO events (timestamp, channel, plate, country, confidence, source, frame_path, plate_path, direction)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    ts,
                    channel,
                    plate,
                    country,
                    confidence,
                    source,
                    frame_path,
                    plate_path,
                    direction,
                ),
            )
            conn.commit()
            self.logger.info(
                "Event saved: %s (%s, country=%s, conf=%.2f, src=%s)",
                plate,
                channel,
                country or "?",
                confidence or 0.0,
                source,
            )
            return cursor.lastrowid

    def fetch_recent(self, limit: int = 100) -> List[sqlite3.Row]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM events ORDER BY datetime(timestamp) DESC LIMIT ?",
                (limit,),
            )
            return cursor.fetchall()

    def fetch_filtered(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        channel: Optional[str] = None,
        plates: Optional[Sequence[str]] = None,
        limit: int = 100,
    ) -> List[sqlite3.Row]:
        filters = []
        params: List[object] = []

        if start:
            filters.append("datetime(timestamp) >= datetime(?)")
            params.append(start)
        if end:
            filters.append("datetime(timestamp) <= datetime(?)")
            params.append(end)
        if channel:
            filters.append("channel = ?")
            params.append(channel)
        if plates:
            placeholders = ",".join("?" for _ in plates)
            filters.append(f"plate IN ({placeholders})")
            params.extend(list(plates))

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        query = f"SELECT * FROM events {where_clause} ORDER BY datetime(timestamp) DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, tuple(params))
            return cursor.fetchall()

    def search_by_plate(
        self,
        plate_fragment: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100,
    ) -> List[sqlite3.Row]:
        filters = ["plate LIKE ?"]
        params: List[object] = [f"%{plate_fragment}%"]

        if start:
            filters.append("datetime(timestamp) >= datetime(?)")
            params.append(start)
        if end:
            filters.append("datetime(timestamp) <= datetime(?)")
            params.append(end)

        where_clause = f"WHERE {' AND '.join(filters)}"
        query = (
            "SELECT * FROM events "
            f"{where_clause} ORDER BY datetime(timestamp) DESC LIMIT ?"
        )
        params.append(limit)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, tuple(params))
            return cursor.fetchall()

    def list_channels(self) -> List[str]:
        with self._connect() as conn:
            cursor = conn.execute("SELECT DISTINCT channel FROM events ORDER BY channel")
            return [row[0] for row in cursor.fetchall()]


class AsyncEventDatabase:
    """Асинхронный доступ к SQLite для фоновых потоков распознавания."""

    def __init__(self, db_path: str = "data/db/anpr.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._initialized = False
        self.logger = get_logger(__name__)

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    plate TEXT NOT NULL,
                    country TEXT,
                    confidence REAL,
                    source TEXT,
                    frame_path TEXT,
                    plate_path TEXT,
                    direction TEXT
                )
                """
            )
            await self._ensure_columns(conn)
            await self._ensure_indexes(conn)
            await conn.commit()
        self._initialized = True

    async def _ensure_columns(self, conn: aiosqlite.Connection) -> None:
        async def _column_exists(name: str) -> bool:
            cursor = await conn.execute("PRAGMA table_info(events)")
            rows = await cursor.fetchall()
            return any(row[1] == name for row in rows)

        if not await _column_exists("frame_path"):
            await conn.execute("ALTER TABLE events ADD COLUMN frame_path TEXT")
        if not await _column_exists("plate_path"):
            await conn.execute("ALTER TABLE events ADD COLUMN plate_path TEXT")
        if not await _column_exists("country"):
            await conn.execute("ALTER TABLE events ADD COLUMN country TEXT")
        if not await _column_exists("direction"):
            await conn.execute("ALTER TABLE events ADD COLUMN direction TEXT")

    @staticmethod
    async def _ensure_indexes(conn: aiosqlite.Connection) -> None:
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_channel ON events(channel)"
        )

    async def insert_event_async(
        self,
        channel: str,
        plate: str,
        confidence: float = 0.0,
        source: str = "",
        timestamp: Optional[str] = None,
        frame_path: Optional[str] = None,
        plate_path: Optional[str] = None,
        country: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> int:
        await self._ensure_schema()
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                (
                    "INSERT INTO events (timestamp, channel, plate, country, confidence, source, frame_path, plate_path, direction)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    ts,
                    channel,
                    plate,
                    country,
                    confidence,
                    source,
                    frame_path,
                    plate_path,
                    direction,
                ),
            )
            await conn.commit()
            self.logger.info(
                "[async] Event saved: %s (%s, country=%s, conf=%.2f, src=%s)",
                plate,
                channel,
                country or "?",
                confidence or 0.0,
                source,
            )
            return cursor.lastrowid
