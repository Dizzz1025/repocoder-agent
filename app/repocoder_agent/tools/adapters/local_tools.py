from __future__ import annotations

from ..registry import ToolRegistry, ToolSpec
from ...repository import RepositoryScanner
from ...skills.loader import SkillLoader


def register_local_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name='scan_repository',
            description='Scan a repository and return summary plus top files.',
            schema={
                'type': 'object',
                'properties': {
                    'repository_path': {'type': 'string'},
                },
                'required': ['repository_path'],
            },
            handler=_scan_repository,
        )
    )
    registry.register(
        ToolSpec(
            name='list_skills',
            description='List skills available in the target repository.',
            schema={
                'type': 'object',
                'properties': {
                    'repository_path': {'type': 'string'},
                },
                'required': ['repository_path'],
            },
            handler=_list_skills,
        )
    )
    registry.register(
        ToolSpec(
            name='show_skill',
            description='Show the contents of a skill in the target repository.',
            schema={
                'type': 'object',
                'properties': {
                    'repository_path': {'type': 'string'},
                    'name': {'type': 'string'},
                },
                'required': ['repository_path', 'name'],
            },
            handler=_show_skill,
        )
    )


def _scan_repository(repository_path: str):
    scanner = RepositoryScanner(repository_path)
    snapshot = scanner.scan()
    return {
        'summary': snapshot.summary.model_dump(mode='json'),
        'top_files': [item.rel_path for item in snapshot.files[:20]],
    }


def _list_skills(repository_path: str):
    loader = SkillLoader(repository_path)
    return {
        'skills': [
            {
                'name': skill.name,
                'path': skill.path,
                'title': skill.title,
                'summary': skill.summary,
                'tags': list(skill.tags),
                'resources': [
                    {
                        'path': resource.relative_path,
                        'type': resource.resource_type,
                    }
                    for resource in skill.resources
                ],
            }
            for skill in loader.list_skills()
        ]
    }


def _show_skill(repository_path: str, name: str):
    loader = SkillLoader(repository_path)
    skill = loader.get_skill(name)
    return {
        'skill': None if skill is None else {
            'name': skill.name,
            'path': skill.path,
            'title': skill.title,
            'summary': skill.summary,
            'tags': list(skill.tags),
            'resources': [
                {
                    'path': resource.relative_path,
                    'type': resource.resource_type,
                }
                for resource in skill.resources
            ],
            'content': skill.content,
        }
    }
