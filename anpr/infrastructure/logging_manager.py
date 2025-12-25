# /anpr/infrastructure/logging_manager.py
"""Централизованная настройка логирования приложения."""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict


class LoggingManager:
    """Создает согласованный стек логирования для GUI, пайплайна и фоновых потоков."""

    DEFAULT_LEVEL = "INFO"
    DEFAULT_FILE = "data/app.log"
    DEFAULT_MAX_BYTES = 1_048_576
    DEFAULT_BACKUP_COUNT = 5

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._configure()

    def _configure(self) -> None:
        level_name = str(self.config.get("level", self.DEFAULT_LEVEL)).upper()
        level = getattr(logging, level_name, logging.INFO)
        log_file = self.config.get("file", self.DEFAULT_FILE)
        max_bytes = int(self.config.get("max_bytes", self.DEFAULT_MAX_BYTES))
        backup_count = int(self.config.get("backup_count", self.DEFAULT_BACKUP_COUNT))

        log_dir = os.path.dirname(log_file) or "."
        os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        logging.getLogger(__name__).debug(
            "Logging configured (level=%s, file=%s, rotation=%s x %s)",
            level_name,
            log_file,
            max_bytes,
            backup_count,
        )


def get_logger(name: str) -> logging.Logger:
    """Утилита для получения именованного логгера."""

    return logging.getLogger(name)
