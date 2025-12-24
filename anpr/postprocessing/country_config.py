#/anpr/postprocessing/country_config.py
"""Загрузка конфигураций форматов номеров из YAML."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


@dataclass
class PlateFormat:
    name: str
    regex: str
    pattern: re.Pattern = field(repr=False)


@dataclass
class CorrectionRules:
    digit_to_letter: Dict[str, str] = field(default_factory=dict)
    letter_to_digit: Dict[str, str] = field(default_factory=dict)
    common_mistakes: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class CountryConfig:
    name: str
    code: str
    priority: int
    formats: List[PlateFormat]
    valid_letters: str
    valid_digits: str
    corrections: CorrectionRules
    stop_words: List[str]
    invalid_sequences: List[str]


class CountryConfigLoader:
    """Читает конфигурации стран из YAML-файлов и готовит их к использованию."""

    def __init__(self, config_dir: str) -> None:
        self.config_dir = Path(config_dir)

    def _load_yaml(self, path: Path) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _compile_format(self, fmt: Dict[str, str]) -> PlateFormat:
        return PlateFormat(
            name=fmt.get("name", "unknown"),
            regex=fmt.get("regex", ""),
            pattern=re.compile(fmt.get("regex", "")),
        )

    def _parse_country(self, data: Dict) -> CountryConfig:
        formats = [self._compile_format(fmt) for fmt in data.get("license_plate_formats", []) if fmt.get("regex")]
        corrections = data.get("corrections", {}) or {}
        return CountryConfig(
            name=data.get("name", ""),
            code=data.get("code", "").upper(),
            priority=int(data.get("priority", 100)),
            formats=formats,
            valid_letters=(data.get("valid_characters", {}) or {}).get("letters", "").upper(),
            valid_digits=(data.get("valid_characters", {}) or {}).get("digits", "0123456789"),
            corrections=CorrectionRules(
                digit_to_letter={(k or "").upper(): (v or "").upper() for k, v in (corrections.get("digit_to_letter") or {}).items()},
                letter_to_digit={(k or "").upper(): (v or "").upper() for k, v in (corrections.get("letter_to_digit") or {}).items()},
                common_mistakes=[
                    {"from": (item.get("from") or "").upper(), "to": (item.get("to") or "").upper()}
                    for item in corrections.get("common_mistakes") or []
                    if item
                ],
            ),
            stop_words=[w.upper() for w in data.get("stop_words", [])],
            invalid_sequences=[seq.upper() for seq in data.get("invalid_sequences", [])],
        )

    def available_configs(self) -> List[Dict[str, str]]:
        """Возвращает метаданные всех доступных шаблонов (код/название)."""

        result: List[Dict[str, str]] = []
        for path in self.config_dir.glob("*.yaml"):
            try:
                data = self._load_yaml(path)
            except FileNotFoundError:
                continue
            result.append(
                {
                    "code": (data.get("code") or path.stem).upper(),
                    "name": data.get("name") or path.stem,
                }
            )
        return sorted(result, key=lambda x: x["code"])

    def load(self, enabled_codes: Optional[Iterable[str]] = None) -> List[CountryConfig]:
        """Загружает только выбранные страны, сохраняя приоритеты."""

        codes = [code.upper() for code in enabled_codes or []]
        country_files = list(self.config_dir.glob("*.yaml"))
        countries: List[CountryConfig] = []

        for path in country_files:
            try:
                data = self._load_yaml(path)
            except FileNotFoundError:
                continue
            code = (data.get("code") or path.stem).upper()
            if codes and code not in codes:
                continue
            countries.append(self._parse_country(data))

        # Если список был пустым или конфиги не найдены, загружаем всё
        if not countries and not codes:
            for path in country_files:
                countries.append(self._parse_country(self._load_yaml(path)))

        return sorted(countries, key=lambda c: c.priority)

    def ensure_dir(self) -> None:
        os.makedirs(self.config_dir, exist_ok=True)
