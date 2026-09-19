from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from app.db import Database
from app.memory.models import ChatTurn, MemoryContext
from app.models.entities import Conversation, ResearchRun
from scripts.manual_invoke_workflow import (
    ModelProgressPrinter,
    _ensure_manual_run,
    _memory_state,
    _model_call_label,
    _record_completed_memory,
    _structured_output_diagnostic,
)


def test_model_call_label_redacts_prompt_content_and_uses_workflow_tag() -> None:
    label = _model_call_label(
        {"name": "ChatOpenAI", "kwargs": {"model_name": "MiniMax-M2.7-highspeed"}},
        ["agent", "assess_coverage"],
    )

    assert label == "node=assess_coverage model=MiniMax-M2.7-highspeed"


@pytest.mark.asyncio
async def test_model_progress_printer_reports_start_and_error(capsys: pytest.CaptureFixture[str]) -> None:
    printer = ModelProgressPrinter()
    run_id = uuid4()

    await printer.on_chat_model_start({"name": "ChatOpenAI"}, [[]], run_id=run_id)
    await printer.on_llm_error(RuntimeError("provider unavailable"), run_id=run_id)

    output = capsys.readouterr().out
    assert "[llm:start]" in output
    assert "[llm:error]" in output
    assert "provider unavailable" in output


def test_structured_diagnostic_reports_invalid_coverage_fields() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "CoverageAssessment",
                "args": {"sufficient": "yes", "confidence": 2},
                "id": "call-1",
            }
        ],
    )

    diagnostic = _structured_output_diagnostic(
        type("Response", (), {"generations": [[type("Generation", (), {"message": message})()]]})(),
        ["assess_coverage"],
    )

    assert diagnostic is not None
    assert diagnostic.startswith("invalid CoverageAssessment:")
    assert "confidence" in diagnostic
    assert "reason" in diagnostic


def test_structured_diagnostic_shows_a_short_plain_text_excerpt_when_no_tool_is_called() -> None:
    message = AIMessage(content="我无法按工具格式回答。")

    diagnostic = _structured_output_diagnostic(
        type("Response", (), {"generations": [[type("Generation", (), {"message": message})()]]})(),
        ["assess_coverage"],
    )

    assert diagnostic is not None
    assert "expected CoverageAssessment" in diagnostic
    assert "text='我无法按工具格式回答。'" in diagnostic


def test_memory_state_serializes_summary_and_recent_turns() -> None:
    state = _memory_state(
        MemoryContext(
            summary="用户关注流匹配。",
            recent_turns=[ChatTurn(role="user", content="继续介绍公式")],
        )
    )

    assert state == {
        "conversation_summary": "用户关注流匹配。",
        "chat_history": [{"role": "user", "content": "继续介绍公式"}],
    }


class _RecordingMemory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

    async def record(self, conversation_id: str, run_id: str, user_query: str, report: str) -> None:
        self.calls.append((conversation_id, run_id, user_query, report))


@pytest.mark.asyncio
async def test_completed_workflow_is_recorded_once_with_final_response() -> None:
    memory = _RecordingMemory()

    await _record_completed_memory(
        memory,  # type: ignore[arg-type]
        {
            "conversation_id": "conversation-1",
            "run_id": "run-1",
            "user_query": "什么是流匹配？",
            "report_content": "流匹配是一类生成建模方法。",
        },
    )

    assert memory.calls == [
        ("conversation-1", "run-1", "什么是流匹配？", "流匹配是一类生成建模方法。")
    ]


@pytest.mark.asyncio
async def test_ensure_manual_run_creates_message_foreign_key_parents(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    await database.initialize()
    try:
        await _ensure_manual_run(
            database,
            conversation_id="conversation-1",
            run_id="run-1",
            graph_thread_id="thread-1",
            user_query="test query",
        )
        async with database.session() as session:
            assert await session.get(Conversation, "conversation-1") is not None
            run_id = await session.scalar(select(ResearchRun.id).where(ResearchRun.id == "run-1"))
        assert run_id == "run-1"
    finally:
        await database.dispose()
