from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ResourceType = Literal["template", "reference", "script", "other"]


@dataclass(frozen=True)
class SkillResource:
    relative_path: str
    absolute_path: str
    resource_type: ResourceType


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    path: str
    title: str
    summary: str
    tags: tuple[str, ...] = ()
    resources: tuple[SkillResource, ...] = ()


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    path: str
    title: str
    summary: str
    content: str
    tags: tuple[str, ...] = ()
    resources: tuple[SkillResource, ...] = ()


class SkillLoader:
    def __init__(self, repo_path: str):
        self.repo_root = Path(repo_path).resolve()
        self.skills_root = self.repo_root / '.repocoder' / 'skills'

    def list_skills(self) -> list[SkillMetadata]:
        if not self.skills_root.exists():
            return []
        skills: list[SkillMetadata] = []
        for skill_dir in sorted(self.skills_root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / 'SKILL.md'
            if not skill_file.exists() or not skill_file.is_file():
                continue
            metadata = self._load_metadata(skill_dir, skill_file)
            skills.append(metadata)
        return skills

    def get_skill(self, name: str) -> SkillDefinition | None:
        skill_dir = self.skills_root / name
        skill_path = skill_dir / 'SKILL.md'
        if not skill_path.exists() or not skill_path.is_file():
            return None
        metadata = self._load_metadata(skill_dir, skill_path)
        content = skill_path.read_text(encoding='utf-8')
        return SkillDefinition(
            name=metadata.name,
            path=metadata.path,
            title=metadata.title,
            summary=metadata.summary,
            content=content,
            tags=metadata.tags,
            resources=metadata.resources,
        )

    def get_skill_resource(self, name: str, relative_path: str) -> str | None:
        skill_dir = self.skills_root / name
        target = (skill_dir / relative_path).resolve()
        if skill_dir.resolve() not in target.parents and target != skill_dir.resolve():
            return None
        if not target.exists() or not target.is_file():
            return None
        return target.read_text(encoding='utf-8')

    def _load_metadata(self, skill_dir: Path, skill_path: Path) -> SkillMetadata:
        title = ''
        summary = ''
        tags: tuple[str, ...] = ()
        metadata_path = skill_dir / 'metadata.json'
        if metadata_path.exists() and metadata_path.is_file():
            try:
                payload = json.loads(metadata_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                title = str(payload.get('title', '')).strip()
                summary = str(payload.get('summary', '')).strip()
                tags_raw = payload.get('tags', [])
                if isinstance(tags_raw, list):
                    tags = tuple(str(item).strip() for item in tags_raw if str(item).strip())
        if not title or not summary:
            parsed_title, parsed_summary = self._read_skill_preview(skill_path)
            title = title or parsed_title
            summary = summary or parsed_summary
        return SkillMetadata(
            name=skill_dir.name,
            path=str(skill_path),
            title=title,
            summary=summary,
            tags=tags,
            resources=self._list_resources(skill_dir),
        )

    def _list_resources(self, skill_dir: Path) -> tuple[SkillResource, ...]:
        resources: list[SkillResource] = []
        for path in sorted(skill_dir.rglob('*')):
            if not path.is_file():
                continue
            if path.name in {'SKILL.md', 'metadata.json'}:
                continue
            relative_path = str(path.relative_to(skill_dir)).replace('\\', '/')
            resources.append(
                SkillResource(
                    relative_path=relative_path,
                    absolute_path=str(path),
                    resource_type=self._resource_type(relative_path),
                )
            )
        return tuple(resources)

    def _resource_type(self, relative_path: str) -> ResourceType:
        if relative_path.startswith('templates/'):
            return 'template'
        if relative_path.startswith('references/'):
            return 'reference'
        if relative_path.startswith('scripts/'):
            return 'script'
        return 'other'

    def _read_skill_preview(self, skill_path: Path) -> tuple[str, str]:
        return self._parse_skill_preview(skill_path.read_text(encoding='utf-8'))

    def _parse_skill_preview(self, content: str) -> tuple[str, str]:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return '', ''

        title = lines[0]
        if title.startswith('#'):
            title = title.lstrip('#').strip()

        summary = ''
        for line in lines[1:]:
            if line.startswith('#'):
                continue
            summary = line
            break

        return title, summary
