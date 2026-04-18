from __future__ import annotations

import json
from pathlib import Path

from repocoder_agent import cli
from repocoder_agent.agent import RepoCoderAgent
from repocoder_agent.models import AgentTaskRequest
from repocoder_agent.skills.loader import SkillLoader


def test_skill_loader_lists_and_reads_skills(tmp_path: Path) -> None:
    skill_dir = tmp_path / '.repocoder' / 'skills' / 'python-bugfix'
    (skill_dir / 'templates').mkdir(parents=True)
    (skill_dir / 'metadata.json').write_text(
        json.dumps(
            {
                'title': 'Python Bugfix',
                'summary': 'Fix Python bugs safely.',
                'tags': ['python', 'bugfix'],
            }
        ),
        encoding='utf-8',
    )
    (skill_dir / 'SKILL.md').write_text(
        '# Python Bugfix\n\nFix Python bugs safely.\n\nPrefer minimal replace patches.\n',
        encoding='utf-8',
    )
    (skill_dir / 'templates' / 'prompt.md').write_text('Prefer minimal changes.\n', encoding='utf-8')

    loader = SkillLoader(str(tmp_path))
    skills = loader.list_skills()
    skill = loader.get_skill('python-bugfix')
    resource = loader.get_skill_resource('python-bugfix', 'templates/prompt.md')

    assert skills
    assert skills[0].title == 'Python Bugfix'
    assert skills[0].summary == 'Fix Python bugs safely.'
    assert skills[0].tags == ('python', 'bugfix')
    assert skills[0].resources[0].relative_path == 'templates/prompt.md'
    assert skill is not None
    assert 'Prefer minimal replace patches.' in skill.content
    assert resource == 'Prefer minimal changes.\n'


def test_cli_skills_list_and_show(tmp_path: Path, capsys) -> None:
    skill_dir = tmp_path / '.repocoder' / 'skills' / 'python-bugfix'
    (skill_dir / 'references').mkdir(parents=True)
    (skill_dir / 'metadata.json').write_text(
        json.dumps(
            {
                'title': 'Python Bugfix',
                'summary': 'Fix Python bugs safely.',
                'tags': ['python', 'bugfix'],
            }
        ),
        encoding='utf-8',
    )
    (skill_dir / 'SKILL.md').write_text(
        '# Python Bugfix\n\nFix Python bugs safely.\n',
        encoding='utf-8',
    )
    (skill_dir / 'references' / 'patterns.md').write_text('Common bugfix patterns.\n', encoding='utf-8')

    exit_code = cli.main(['skills', str(tmp_path), 'list'])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload['skills'][0]['name'] == 'python-bugfix'
    assert payload['skills'][0]['title'] == 'Python Bugfix'
    assert payload['skills'][0]['summary'] == 'Fix Python bugs safely.'
    assert payload['skills'][0]['resources'][0]['path'] == 'references/patterns.md'

    exit_code = cli.main(['skills', str(tmp_path), 'show', 'python-bugfix'])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload['skill']['title'] == 'Python Bugfix'
    assert 'Fix Python bugs safely' in payload['skill']['content']
    assert payload['skill']['resources'][0]['path'] == 'references/patterns.md'


def test_plan_mode_can_include_skill_context(tmp_path: Path) -> None:
    skill_dir = tmp_path / '.repocoder' / 'skills' / 'python-bugfix'
    skill_dir.mkdir(parents=True)
    (skill_dir / 'SKILL.md').write_text('# Python Bugfix\n\nPrefer minimal replace patches.\n', encoding='utf-8')
    (tmp_path / 'check.py').write_text('value = "bad"\nassert value == "good"\n', encoding='utf-8')

    task = AgentTaskRequest(
        repository_path=str(tmp_path),
        goal='in check.py replace `"bad"` with `"good"`',
        commands=['python check.py'],
        mode='plan',
        skill='python-bugfix',
        auto_fix=False,
    )
    result = RepoCoderAgent(task).run()

    assert result.success is True
    assert result.mode == 'plan'
    assert result.proposed_patches
