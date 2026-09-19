"""Interactively smoke-test the real LangGraph workflow with ``ainvoke``.

This script uses configured MiniMax, DashScope, Chroma, and the durable SQLite
checkpointer. It can therefore consume model credits and, after explicit input,
perform arXiv search and PDF import. It never auto-approves either action.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from time import perf_counter
from typing import Any, cast
from uuid import UUID, uuid4

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.types import Command
from pydantic import BaseModel, ValidationError
from sqlalchemy import select

from app.config import get_settings
from app.db import Database
from app.graph.checkpointer import open_sqlite_checkpointer
from app.graph.factory import build_configured_research_graph
from app.llm.minimax import create_minimax_chat_model
from app.memory import (
    InMemorySummaryCache,
    MemoryContext,
    MemoryManager,
    MemorySummarizer,
    SQLiteMessageStore,
)
from app.models.entities import Conversation, ResearchRun
from app.models.schemas import ArxivSearchPlan, CoverageAssessment, GroundedReportDraft, QueryIntent

DEFAULT_QUERY = "介绍一下生成模型"
DEFAULT_CONVERSATION_ID = "manual-smoke-conversation"


class ModelProgressPrinter(AsyncCallbackHandler):
    """Print nested LangChain model calls that task-level graph streaming hides."""

    def __init__(self) -> None:
        self._started_at: dict[UUID, float] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del messages, parent_run_id, metadata, kwargs
        self._started_at[run_id] = perf_counter()
        label = _model_call_label(serialized, tags)
        print(f"  [llm:start] {label} | run={str(run_id)[:8]}")

    async def on_llm_end(
        self, response: Any, *, run_id: UUID, tags: list[str] | None = None, **__: Any
    ) -> None:
        elapsed = perf_counter() - self._started_at.pop(run_id, perf_counter())
        print(f"  [llm:end]   run={str(run_id)[:8]} | {elapsed:.2f}s")
        diagnostic = _structured_output_diagnostic(response, tags)
        if diagnostic is not None:
            print(f"  [structured] {diagnostic}")

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **__: Any) -> None:
        elapsed = perf_counter() - self._started_at.pop(run_id, perf_counter())
        print(f"  [llm:error] run={str(run_id)[:8]} | {elapsed:.2f}s | {type(error).__name__}: {error}")


def _model_call_label(serialized: dict[str, Any], tags: list[str] | None) -> str:
    """Keep live diagnostics useful without printing prompt contents or credentials."""
    name = serialized.get("name")
    model = serialized.get("kwargs", {}).get("model_name") if isinstance(serialized.get("kwargs"), dict) else None
    workflow_tags = [tag for tag in tags or [] if tag in {"classify_query", "assess_coverage", "build_arxiv_query", "generate_report"}]
    context = workflow_tags[0] if workflow_tags else "agent"
    return f"node={context} model={model or name or 'chat_model'}"


def _structured_output_diagnostic(response: Any, tags: list[str] | None) -> str | None:
    """Explain ToolStrategy failures without printing prompts or full model output."""
    schema = _schema_for_tags(tags)
    if schema is None:
        return None
    message = _first_ai_message(response)
    if message is None:
        return "no AI message available for structured-output inspection"
    tool_calls = message.tool_calls
    matching_calls = [call for call in tool_calls if call.get("name") == schema.__name__]
    if not matching_calls:
        invalid_calls = getattr(message, "invalid_tool_calls", [])
        if invalid_calls:
            names = [call.get("name") for call in invalid_calls if isinstance(call, dict)]
            return f"invalid tool calls received: {names or 'unnamed'}"
        content = " ".join(str(message.content).split())
        excerpt = f"; text={content[:240]!r}" if content else ""
        return (
            f"expected {schema.__name__} tool call; received "
            f"{[call.get('name') for call in tool_calls] or 'none'}{excerpt}"
        )
    if len(matching_calls) > 1:
        return f"received {len(matching_calls)} {schema.__name__} tool calls; exactly one is required"
    try:
        schema.model_validate(matching_calls[0].get("args", {}))
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors(include_input=False)
        )
        return f"invalid {schema.__name__}: {details}"
    return f"valid {schema.__name__} tool call"


def _schema_for_tags(tags: list[str] | None) -> type[BaseModel] | None:
    mapping: dict[str, type[BaseModel]] = {
        "classify_query": QueryIntent,
        "assess_coverage": CoverageAssessment,
        "build_arxiv_query": ArxivSearchPlan,
        "generate_report": GroundedReportDraft,
    }
    return next((mapping[tag] for tag in tags or [] if tag in mapping), None)


def _first_ai_message(response: Any) -> AIMessage | None:
    generations = getattr(response, "generations", None)
    if not isinstance(generations, list) or not generations or not generations[0]:
        return None
    message = getattr(generations[0][0], "message", None)
    return message if isinstance(message, AIMessage) else None


async def main(query: str | None, thread_id: str | None, conversation_id: str | None = None) -> None:
    cast(Any, sys.stdout).reconfigure(encoding="utf-8")
    settings = get_settings()
    _validate_credentials(settings)
    database = Database(settings.database_path)
    await database.initialize()
    memory = MemoryManager(
        SQLiteMessageStore(database),
        InMemorySummaryCache(),
        MemorySummarizer(create_minimax_chat_model(settings)),
        tail_keep=4,
        tail_count_limit=4,
    )
    try:
        async with open_sqlite_checkpointer(settings) as checkpointer:
            graph = build_configured_research_graph(settings, checkpointer=checkpointer)
            if thread_id is None:
                run_id = str(uuid4())
                thread_id = f"manual-{run_id}"
                active_conversation_id = conversation_id or DEFAULT_CONVERSATION_ID
                user_query = query or DEFAULT_QUERY
                await _ensure_manual_run(
                    database,
                    conversation_id=active_conversation_id,
                    run_id=run_id,
                    graph_thread_id=thread_id,
                    user_query=user_query,
                )
                memory_context = await memory.inject(active_conversation_id)
                initial_state = {
                    "run_id": run_id,
                    "conversation_id": active_conversation_id,
                    "workspace_id": "default",
                    "user_query": user_query,
                    "status": "started",
                    **_memory_state(memory_context),
                }
            else:
                initial_state = {}
            config = {
                "configurable": {"thread_id": thread_id},
                "callbacks": [ModelProgressPrinter()],
            }
            timings: list[tuple[str, float, str]] = []
            if initial_state:
                print(f"thread_id={thread_id}")
                print(f"conversation_id={initial_state['conversation_id']}")
                print(f"memory=summary:{len(str(initial_state['conversation_summary']))} chars, "
                      f"recent_turns:{len(cast(list[object], initial_state['chat_history']))}")
                print(f"query={initial_state['user_query']}")
                result, payload = await _stream_invocation(
                    graph, initial_state, config=cast(Any, config), timings=timings
                )
            else:
                snapshot = await graph.aget_state(cast(Any, config))
                result = dict(snapshot.values)
                payload = _snapshot_interrupt_payload(snapshot)
                if payload is None:
                    raise ValueError(f"thread_id={thread_id!r} has no pending human approval.")
                print(f"thread_id={thread_id} (resuming pending approval)")
                print(f"query={result.get('user_query')}")

            while payload is not None:
                print("\n--- Human approval required ---")
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                command = await _approval_command(payload)
                if command is None:
                    print("Workflow remains paused. Re-run with the printed thread_id to resume later.")
                    _print_timing_summary(timings)
                    return
                result, payload = await _stream_invocation(
                    graph, command, config=cast(Any, config), timings=timings
                )

        await _record_completed_memory(memory, result)
        _print_result(result)
        _print_timing_summary(timings)
    finally:
        await database.dispose()


def _memory_state(context: MemoryContext) -> dict[str, object]:
    """Convert bounded durable memory into serializable LangGraph input state."""
    return {
        "conversation_summary": context.summary,
        "chat_history": [turn.model_dump() for turn in context.recent_turns],
    }


async def _ensure_manual_run(
    database: Database,
    *,
    conversation_id: str,
    run_id: str,
    graph_thread_id: str,
    user_query: str,
) -> None:
    """Create the SQLite parents required by durable messages for a new manual run."""
    async with database.session() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None:
            session.add(Conversation(id=conversation_id, title="Manual smoke conversation"))
        existing_run = await session.scalar(select(ResearchRun.id).where(ResearchRun.id == run_id))
        if existing_run is None:
            session.add(
                ResearchRun(
                    id=run_id,
                    conversation_id=conversation_id,
                    graph_thread_id=graph_thread_id,
                    status="running",
                    user_query=user_query,
                )
            )
        await session.commit()


async def _record_completed_memory(memory: MemoryManager, result: dict[str, Any]) -> None:
    """Persist a finished turn once; interrupted runs intentionally remain unrecorded."""
    report = result.get("report_content")
    conversation_id = result.get("conversation_id")
    run_id = result.get("run_id")
    query = result.get("user_query")
    if not all(isinstance(value, str) for value in (report, conversation_id, run_id, query)):
        return
    await memory.record(
        cast(str, conversation_id),
        cast(str, run_id),
        cast(str, query),
        cast(str, report),
    )


async def _stream_invocation(
    graph: Any,
    workflow_input: dict[str, object] | Command,
    *,
    config: dict[str, Any],
    timings: list[tuple[str, float, str]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run one graph segment while printing node lifecycle events and timings."""
    started_at: dict[str, float] = {}
    async for event in graph.astream(workflow_input, config=config, stream_mode="tasks"):
        if not isinstance(event, dict):
            continue
        task_id = event.get("id")
        node_name = event.get("name")
        if not isinstance(task_id, str) or not isinstance(node_name, str):
            continue
        if "result" not in event:
            started_at[task_id] = perf_counter()
            print(f"[start] {node_name}")
            continue
        elapsed = perf_counter() - started_at.pop(task_id, perf_counter())
        error = event.get("error")
        outcome = "error" if error else "interrupted" if event.get("interrupts") else "completed"
        timings.append((node_name, elapsed, outcome))
        print(f"[end]   {node_name} | {elapsed:.2f}s | {outcome}")
        if error:
            print(f"        error: {error}")

    snapshot = await graph.aget_state(config)
    values = dict(snapshot.values)
    return values, _snapshot_interrupt_payload(snapshot)


def _snapshot_interrupt_payload(snapshot: Any) -> dict[str, Any] | None:
    """Read the pending interrupt from a checkpointer snapshot after streaming ends."""
    for task in snapshot.tasks:
        interrupts = getattr(task, "interrupts", ())
        for interrupt in interrupts:
            value = getattr(interrupt, "value", None)
            if isinstance(value, dict):
                return value
    return None


async def _approval_command(payload: dict[str, Any]) -> Command | None:
    approval_type = payload.get("approval_type")
    if approval_type == "search_arxiv":
        response = await _read_input("Type approve to run arXiv search; stop to pause: ")
        if response is not None and response.casefold() in {"approve", "approval"}:
            return Command(resume={"decision": "approve"})
        if response not in {None, "", "stop"}:
            print("未识别输入；请输入 approve（或 approval）以同意，或 stop 暂停。")
        return None
    if approval_type == "import_papers":
        offered_ids: list[str] = []
        candidates = payload.get("candidates", [])
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                paper_id = item.get("paper_id")
                if isinstance(paper_id, str):
                    offered_ids.append(paper_id)
        print(f"可选 paper_id: {', '.join(offered_ids) or '（没有候选论文）'}")
        response = await _read_input("paper_id list to import, or stop to pause: ")
        if response is None:
            return None
        if response == "stop":
            return None
        selected_ids = [item.strip() for item in response.split(",") if item.strip()]
        if not selected_ids:
            print("未选择论文；工作流保持暂停。")
            return None
        return Command(resume={"decision": "select", "selected_paper_ids": selected_ids})
    raise RuntimeError(f"Unexpected interrupt approval_type: {approval_type!r}")


async def _read_input(prompt: str) -> str | None:
    try:
        return (await asyncio.to_thread(input, prompt)).strip()
    except EOFError:
        print("\nNo interactive stdin is available; leaving the workflow paused.")
        return None


def _validate_credentials(settings: Any) -> None:
    if settings.minimax_api_key is None:
        raise RuntimeError("MINIMAX_API_KEY is required; add it to .env before running this script.")
    if settings.dashscope_api_key is None:
        raise RuntimeError("DASHSCOPE_API_KEY is required; add it to .env before running this script.")


def _print_result(result: dict[str, Any]) -> None:
    print(f"\nstatus={result.get('status')}")
    report_content = result.get("report_content")
    if isinstance(report_content, str):
        print("\n--- Generated report ---\n")
        print(report_content)
        return
    print("\n--- Final state (report branch may not be implemented yet) ---")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def _print_timing_summary(timings: list[tuple[str, float, str]]) -> None:
    if not timings:
        return
    total_seconds = sum(elapsed for _, elapsed, _ in timings)
    print("\n--- Node timing summary ---")
    for node_name, elapsed, outcome in timings:
        print(f"{node_name:30} {elapsed:7.2f}s  {outcome}")
    print(f"{'total':30} {total_seconds:7.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the real Paper Assistant workflow manually.")
    parser.add_argument("--query", help=f"Research question to execute (default: {DEFAULT_QUERY}).")
    parser.add_argument("--thread-id", help="Resume a paused manual workflow with this thread_id.")
    parser.add_argument(
        "--conversation-id",
        help=(
            "Durable conversation ID for a new manual run "
            f"(default: {DEFAULT_CONVERSATION_ID})."
        ),
    )
    arguments = parser.parse_args()
    if arguments.query and arguments.thread_id:
        parser.error("--query and --thread-id cannot be used together.")
    if arguments.conversation_id and arguments.thread_id:
        parser.error("--conversation-id cannot be used when resuming a thread.")
    asyncio.run(main(arguments.query, arguments.thread_id, arguments.conversation_id))
