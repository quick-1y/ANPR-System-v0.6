#/anpr/postprocessing/validator.py
"""Постпроцессинг и валидация номеров с поддержкой плагинов стран."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .country_config import CountryConfig, CountryConfigLoader


@dataclass
class PlatePostprocessResult:
    original: str
    normalized: str
    plate: str
    country: Optional[str]
    is_valid: bool
    format_name: Optional[str] = None


class PlatePostProcessor:
    """Выполняет коррекцию и валидацию номеров после OCR."""

    def __init__(self, config_loader: CountryConfigLoader, enabled_countries: Optional[Iterable[str]] = None) -> None:
        self.loader = config_loader
        self.countries: List[CountryConfig] = self.loader.load(enabled_countries)

    @staticmethod
    def _normalize(raw: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-zА-ЯЁ]+", "", raw or "")
        normalized = cleaned.upper().replace("Ё", "Е")
        return normalized

    def _apply_corrections(self, text: str, country: CountryConfig) -> str:
        corrected = text
        for mistake in country.corrections.common_mistakes:
            src = mistake.get("from", "")
            dst = mistake.get("to", "")
            if src:
                corrected = corrected.replace(src, dst)
        for src, dst in country.corrections.digit_to_letter.items():
            corrected = corrected.replace(src, dst)
        for src, dst in country.corrections.letter_to_digit.items():
            corrected = corrected.replace(src, dst)
        return corrected

    def _valid_characters(self, text: str, country: CountryConfig) -> bool:
        allowed = set(country.valid_digits + country.valid_letters)
        return all(ch in allowed for ch in text)

    @staticmethod
    def _contains_invalid_sequences(text: str, sequences: List[str]) -> bool:
        return any(seq and seq in text for seq in sequences)

    def _match_country(self, text: str, country: CountryConfig) -> Optional[str]:
        for fmt in country.formats:
            if fmt.pattern.fullmatch(text):
                return fmt.name
        return None

    def _check_stop_words(self, text: str, stop_words: List[str]) -> bool:
        return any(text == stop_word for stop_word in stop_words)

    def _variants(self, normalized: str, country: CountryConfig) -> List[str]:
        variants = [normalized]
        corrected = self._apply_corrections(normalized, country)
        if corrected and corrected not in variants:
            variants.append(corrected)
        return variants

    def process(self, raw_text: str) -> PlatePostprocessResult:
        normalized = self._normalize(raw_text)
        if not self.countries:
            return PlatePostprocessResult(raw_text, normalized, normalized, None, True, None)

        for country in self.countries:
            for candidate in self._variants(normalized, country):
                if not candidate:
                    continue

                if self._check_stop_words(candidate, country.stop_words):
                    return PlatePostprocessResult(raw_text, normalized, "", country.code, False, None)

                if self._contains_invalid_sequences(candidate, country.invalid_sequences):
                    continue

                if country.valid_letters and not self._valid_characters(candidate, country):
                    continue

                format_name = self._match_country(candidate, country)
                if format_name:
                    return PlatePostprocessResult(raw_text, normalized, candidate, country.code, True, format_name)

        return PlatePostprocessResult(raw_text, normalized, normalized, None, False, None)
