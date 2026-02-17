"""
Archetype library — Layer 1 of the two-layer classification system.

A curated lookup table of market "templates" with pre-assigned insider risk
scores.  When a new market arrives, we first check if it matches an existing
archetype (cheap string matching).  Only if no archetype matches do we call
the LLM (Layer 2).
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import ARCHETYPES_PATH

logger = logging.getLogger(__name__)


@dataclass
class ArchetypeMatch:
    """Result of matching a market against the archetype library."""
    archetype_id: str
    score: int
    reasoning: str
    info_holders: list[str]
    confidence: str  # "high" if pattern match is strong


class ArchetypeLibrary:
    """
    Manages a JSON file of market archetypes.

    Each archetype has:
      - score: insider risk score 1-10
      - reasoning: why this score
      - info_holders: who has advance knowledge
      - patterns: list of lowercase substring patterns to match against
                  market titles/descriptions
    """

    def __init__(self, path: Path = ARCHETYPES_PATH):
        self.path = path
        self.archetypes: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self.archetypes = json.loads(self.path.read_text())
            logger.info("Loaded %d archetypes from %s", len(self.archetypes), self.path)
        else:
            self.archetypes = {}
            logger.warning("No archetypes file at %s", self.path)

    def save(self) -> None:
        self.path.write_text(json.dumps(self.archetypes, indent=2))
        logger.info("Saved %d archetypes to %s", len(self.archetypes), self.path)

    def match(self, title: str, description: str = "") -> Optional[ArchetypeMatch]:
        """
        Try to match a market title/description to an existing archetype.

        Returns ArchetypeMatch if a match is found, None otherwise.
        Matches by checking if any archetype pattern appears as a substring
        in the title or description.
        """
        text = f"{title} {description}".lower()

        best_match: Optional[tuple[str, dict, int]] = None  # (id, archetype, match_count)

        for arch_id, arch in self.archetypes.items():
            patterns = arch.get("patterns", [])
            match_count = sum(1 for p in patterns if p in text)
            if match_count > 0:
                if best_match is None or match_count > best_match[2]:
                    best_match = (arch_id, arch, match_count)

        if best_match is None:
            return None

        arch_id, arch, match_count = best_match
        total_patterns = len(arch.get("patterns", []))
        confidence = "high" if match_count >= 2 or match_count / max(total_patterns, 1) > 0.3 else "medium"

        return ArchetypeMatch(
            archetype_id=arch_id,
            score=arch["score"],
            reasoning=arch["reasoning"],
            info_holders=arch.get("info_holders", []),
            confidence=confidence,
        )

    def add_archetype(
        self,
        archetype_id: str,
        score: int,
        reasoning: str,
        patterns: list[str],
        info_holders: Optional[list[str]] = None,
    ) -> None:
        """Add a new archetype (e.g. from LLM suggestions after classification)."""
        self.archetypes[archetype_id] = {
            "score": score,
            "reasoning": reasoning,
            "info_holders": info_holders or [],
            "patterns": [p.lower() for p in patterns],
        }
        self.save()
        logger.info("Added archetype '%s' (score=%d)", archetype_id, score)

    def list_archetypes(self) -> list[tuple[str, int, str]]:
        """Return (id, score, reasoning) for all archetypes."""
        return [
            (aid, a["score"], a["reasoning"])
            for aid, a in sorted(self.archetypes.items(), key=lambda x: x[1]["score"])
        ]
