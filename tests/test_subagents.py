from __future__ import annotations

from repocoder_agent.agent import RepoCoderAgent
from repocoder_agent.agents.patch_reviewer_agent import PatchReviewerAgent
from repocoder_agent.agents.planner_agent import PlannerAgent
from repocoder_agent.agents.repo_explorer_agent import RepoExplorerAgent
from repocoder_agent.memory.graph_builder import RepositoryGraphBuilder
from repocoder_agent.memory.graph_store import RepositoryGraphStore
from repocoder_agent.memory.history_store import RepositoryHistoryStore
from repocoder_agent.planner import TaskPlanner
from repocoder_agent.repository import RepositoryScanner
from repocoder_agent.retrieval.hybrid_retriever import HybridRetriever
from repocoder_agent.selectors.patch_selector import PatchSelector
from repocoder_agent.policies.uncertainty_gate import UncertaintyGate
from repocoder_agent.critics.patch_critic import PatchCritic
from repocoder_agent.models import AgentTaskRequest, PatchInstruction


def test_repo_explorer_agent_produces_relevant_files(tmp_path) -> None:
    (tmp_path / 'math_utils.py').write_text('def add(a, b):\n    return a + b\n', encoding='utf-8')
    scanner = RepositoryScanner(str(tmp_path))
    explorer = RepoExplorerAgent(
        scanner=scanner,
        graph_store=RepositoryGraphStore(str(tmp_path)),
        graph_builder=RepositoryGraphBuilder(),
        history_store=RepositoryHistoryStore(str(tmp_path)),
        retriever=HybridRetriever(),
    )

    result = explorer.explore(goal='fix add function', top_k_files=3)

    assert result.retrieval_result.relevant_files
    assert result.retrieval_result.relevant_files[0].file_path == 'math_utils.py'


def test_planner_agent_wraps_task_planner(tmp_path) -> None:
    planner_agent = PlannerAgent(TaskPlanner(start_dir=str(tmp_path)))
    result = planner_agent.create_plan(
        goal='fix parser bug',
        relevant_files=[],
        commands=['python -m pytest -q'],
        patches=[],
        auto_fix=True,
    )

    assert result.plan_steps


def test_patch_reviewer_agent_selects_candidate(tmp_path) -> None:
    selector = PatchSelector(
        uncertainty_gate=UncertaintyGate(),
        patch_critic=PatchCritic(),
    )
    reviewer = PatchReviewerAgent(selector)
    scanner = RepositoryScanner(str(tmp_path))
    (tmp_path / 'check.py').write_text('value = "bad"\nassert value == "good"\n', encoding='utf-8')
    snapshot = scanner.scan()
    relevant_files = HybridRetriever().retrieve(
        snapshot=snapshot,
        goal='fix value',
        graph=RepositoryGraphBuilder().build_from_snapshot(snapshot),
        top_k=3,
    )
    result = reviewer.review_candidates(
        candidates=[(
            'rule',
            PatchInstruction(
                file_path='check.py',
                operation='replace',
                find_text='"bad"',
                replace_text='"good"',
            ),
        )],
        relevant_files=relevant_files,
        snapshot=snapshot,
        goal='fix value',
        phase='initial',
    )

    assert result.selection.selected_patch is not None
    assert result.selection.selected_patch.file_path == 'check.py'


def test_repo_coder_agent_still_runs_with_subagents(tmp_path) -> None:
    (tmp_path / 'check.py').write_text('value = "bad"\nassert value == "good"\n', encoding='utf-8')
    task = AgentTaskRequest(
        repository_path=str(tmp_path),
        goal='in check.py replace `"bad"` with `"good"`',
        commands=['python check.py'],
        auto_fix=False,
    )
    result = RepoCoderAgent(task).run()

    assert result.success is True
