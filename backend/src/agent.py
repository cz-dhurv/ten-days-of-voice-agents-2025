
"""
Day 3 — Health & Wellness Companion (Todoist MCP + Zapier webhook)
- Tools attached to the Agent (compatible with livekit-agents versions that don't accept tools on LLM)
- Persist check-ins to backend/wellness_log.json
- Todoist integration via MCP server (todoist-mcp)
- Zapier integration via webhook (HTTP POST)
"""

import logging
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Annotated, Tuple, Dict

from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
    metrics,
    MetricsCollectedEvent,
    RunContext,
    function_tool,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# local MCP helper (backend/mcp_clients.py)
from mcp_clients import get_mcp_client

logger = logging.getLogger("wellness_agent")
logging.basicConfig(level=logging.INFO)

load_dotenv(".env.local")

# Zapier webhook (HTTP) read from env
ZAPIER_WEBHOOK_URL = os.getenv("ZAPIER_WEBHOOK_URL")  # keep as fallback


# -------------------------
# Data model
# -------------------------
@dataclass
class CheckInEntry:
    timestamp_iso: str
    date_str: str
    mood_text: str
    mood_scale: Optional[int]
    energy: Optional[str]
    stress: Optional[str]
    objectives: List[str]
    summary: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp_iso": self.timestamp_iso,
            "date_str": self.date_str,
            "mood_text": self.mood_text,
            "mood_scale": self.mood_scale,
            "energy": self.energy,
            "stress": self.stress,
            "objectives": self.objectives,
            "summary": self.summary,
        }

    @staticmethod
    def from_dict(d: dict) -> "CheckInEntry":
        return CheckInEntry(
            timestamp_iso=d.get("timestamp_iso", ""),
            date_str=d.get("date_str", ""),
            mood_text=d.get("mood_text", ""),
            mood_scale=d.get("mood_scale"),
            energy=d.get("energy"),
            stress=d.get("stress"),
            objectives=d.get("objectives", []) or [],
            summary=d.get("summary"),
        )


@dataclass
class Userdata:
    session_id: str
    current_entry: Optional[CheckInEntry] = None


# -------------------------
# Storage helpers
# -------------------------
def backend_root() -> str:
    base = os.path.dirname(__file__)  # backend/src
    return os.path.abspath(os.path.join(base, ".."))


def wellness_log_path() -> str:
    path = os.path.join(backend_root(), "wellness_log.json")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
    return path


def load_wellness_log() -> List[CheckInEntry]:
    path = wellness_log_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return [CheckInEntry.from_dict(item) for item in data if isinstance(item, dict)]
    except Exception as e:
        logger.exception("Failed to read wellness log: %s", e)
        return []


def append_wellness_entry(entry: CheckInEntry) -> str:
    path = wellness_log_path()
    log = load_wellness_log()
    log.append(entry)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in log], f, indent=2, ensure_ascii=False)
        logger.info("Appended wellness entry to %s", path)
        return path
    except Exception as e:
        logger.exception("Failed to write wellness log: %s", e)
        raise


# -------------------------
# Utilities
# -------------------------
def summarize_entry(entry: CheckInEntry) -> str:
    obj_part = ""
    if entry.objectives:
        obj_part = " Goals: " + "; ".join(entry.objectives[:3])
    scale_part = f" (mood {entry.mood_scale}/10)" if entry.mood_scale else ""
    stress_part = f" Stress: {entry.stress}." if entry.stress else ""
    return f"On {entry.date_str} you said: \"{entry.mood_text}\"{scale_part}. Energy: {entry.energy}. {stress_part}{obj_part}"


def weekly_trend_summary(entries: List[CheckInEntry], days: int = 7) -> str:
    if not entries:
        return "No previous check-ins found."
    cutoff = datetime.utcnow() - timedelta(days=days)
    recent = [e for e in entries if datetime.fromisoformat(e.timestamp_iso) >= cutoff]
    if not recent:
        return f"No entries in the last {days} days."
    scales = [e.mood_scale for e in recent if isinstance(e.mood_scale, int)]
    avg = None
    if scales:
        avg = sum(scales) / len(scales)
    objectives_count = sum(1 for e in recent if e.objectives)
    return f"In the last {len(recent)} check-ins, you recorded objectives on {objectives_count} days." + (f" Average mood: {avg:.1f}/10." if avg else "")


# -------------------------
# Function tools (attached to Agent)
# -------------------------
@function_tool
async def start_checkin(ctx: RunContext[Userdata]) -> str:
    now = datetime.utcnow()
    entry = CheckInEntry(
        timestamp_iso=now.isoformat(),
        date_str=now.strftime("%Y-%m-%d"),
        mood_text="",
        mood_scale=None,
        energy=None,
        stress=None,
        objectives=[],
        summary=None,
    )
    ctx.userdata.current_entry = entry
    history = load_wellness_log()
    ref_line = ""
    if history:
        last = history[-1]
        ref_line = f" Last time you said: \"{last.mood_text}\" on {last.date_str}. How does today compare?"
    return "Hi — I'm your wellness companion. How are you feeling today? You can say a sentence and, optionally, a mood score from 1 to 10." + ref_line


@function_tool
async def record_mood(
    ctx: RunContext[Userdata],
    mood_text: Annotated[str, Field(description="Short description of mood")],
    mood_scale: Annotated[Optional[int], Field(description="Optional mood scale 1-10")] = None,
) -> str:
    entry = ctx.userdata.current_entry
    if not entry:
        return "We haven't started a check-in yet. Say 'start check-in' to begin."
    entry.mood_text = mood_text.strip()
    if mood_scale is not None:
        entry.mood_scale = int(max(1, min(10, mood_scale)))
    logger.info("Recorded mood: %s (%s)", entry.mood_text, entry.mood_scale)
    return "Thanks — noted. What's your energy like right now (low / medium / high)?"


@function_tool
async def record_energy(
    ctx: RunContext[Userdata],
    energy: Annotated[str, Field(description="low / medium / high")],
) -> str:
    entry = ctx.userdata.current_entry
    if not entry:
        return "No active check-in. Please start one first."
    entry.energy = energy.strip().lower()
    logger.info("Recorded energy: %s", entry.energy)
    return "Got it. Is anything stressing you out at the moment? (briefly) If not, say 'no'."


@function_tool
async def record_stress(
    ctx: RunContext[Userdata],
    stress: Annotated[Optional[str], Field(description="Short stress notes (or 'no')")] = None,
) -> str:
    entry = ctx.userdata.current_entry
    if not entry:
        return "No active check-in. Please start one first."
    entry.stress = (stress.strip() if stress else "")
    logger.info("Recorded stress: %s", entry.stress)
    return "Understood. What are 1 to 3 small objectives you'd like to accomplish today? You can list them now."


@function_tool
async def add_objectives(
    ctx: RunContext[Userdata],
    objectives: Annotated[List[str], Field(description="List of short objectives (1-3)")],
) -> str:
    entry = ctx.userdata.current_entry
    if not entry:
        return "No active check-in. Please start one first."
    cleaned = [o.strip() for o in objectives if o and o.strip()]
    remaining = max(0, 3 - len(entry.objectives))
    entry.objectives.extend(cleaned[:remaining])
    logger.info("Added objectives: %s", entry.objectives)
    return f"Added {len(cleaned[:remaining])} objective(s). Would you like a short suggestion for achieving them (yes/no)?"


@function_tool
async def give_suggestion(ctx: RunContext[Userdata], want_suggestion: Annotated[bool, Field(description="yes/no")] = True) -> str:
    entry = ctx.userdata.current_entry
    if not entry:
        return "No active check-in."
    objs = entry.objectives or []
    energy = (entry.energy or "medium").lower()
    tips = []
    if energy == "low":
        tips.append("Break tasks into 10-minute chunks and start with the smallest step.")
        tips.append("Try a 5-minute walk or short rest between tasks.")
    elif energy == "high":
        tips.append("Tackle the most important task while momentum is high.")
        tips.append("Take short breaks to avoid burnout.")
    else:
        tips.append("Start with one achievable task and celebrate the small win.")
    for o in objs[:2]:
        tips.append(f"For '{o}', split it into two clear steps.")
    suggestion = " ".join(tips[:3])
    return f"Here are a few simple suggestions: {suggestion} Would you like me to save this check-in now?"


@function_tool
async def complete_checkin(
    ctx: RunContext[Userdata],
    create_todoist: Annotated[bool, Field(description="Create Todoist tasks?")] = False,
    trigger_zap: Annotated[bool, Field(description="Trigger Zapier webhook?")] = False,
) -> str:
    entry = ctx.userdata.current_entry
    if not entry:
        return "No active check-in to complete. Start a check-in first."
    if not entry.mood_text:
        return "I don't have your mood yet. How are you feeling?"
    entry.summary = summarize_entry(entry)
    # Save locally
    try:
        path = append_wellness_entry(entry)
    except Exception as e:
        logger.exception("Failed to save wellness entry: %s", e)
        return "There was an error saving your check-in — please try again later."
    responses = [f"Thanks — I've saved your check-in ({path}). Here's a quick recap: {entry.summary}"]
    # MCP Todoist
    if create_todoist and entry.objectives:
        try:
            session, tools = await get_mcp_client("todoist")
            created = 0
            # If a tool named 'create_task' exists use it, otherwise call first available tool
            if isinstance(tools, dict) and "create_task" in tools:
                for o in entry.objectives:
                    await session.call_tool("create_task", {"content": o})
                    created += 1
            else:
                tool_name = next(iter(tools.keys())) if tools else None
                if tool_name:
                    for o in entry.objectives:
                        await session.call_tool(tool_name, {"content": o})
                        created += 1
            responses.append(f"Created {created} Todoist task(s).")
        except Exception as e:
            logger.exception("Todoist MCP failed: %s", e)
            responses.append("Todoist integration failed or was not configured.")
    # Zapier webhook via HTTP (fallback)
    if trigger_zap:
        try:
            if ZAPIER_WEBHOOK_URL:
                import requests
                resp = requests.post(ZAPIER_WEBHOOK_URL, json=entry.to_dict(), timeout=10)
                if 200 <= resp.status_code < 300:
                    responses.append("Zapier webhook triggered.")
                else:
                    responses.append("Zapier webhook failed (non-2xx response).")
            else:
                responses.append("Zapier webhook URL not configured.")
        except Exception as e:
            logger.exception("Zapier webhook failed: %s", e)
            responses.append("Zapier trigger failed due to an error.")
    # Clear session entry
    ctx.userdata.current_entry = None
    # Weekly insight
    insight = weekly_trend_summary(load_wellness_log(), days=7)
    responses.append(insight)
    return " ".join(responses)


@function_tool
async def get_wellness_history(ctx: RunContext[Userdata], limit: Annotated[int, Field(description="Number of items to return")] = 3) -> str:
    entries = load_wellness_log()
    if not entries:
        return "No wellness history found."
    selected = entries[-limit:]
    return "\n".join([summarize_entry(e) for e in selected])


@function_tool
async def weekly_summary(ctx: RunContext[Userdata], days: Annotated[int, Field(description="Days window")] = 7) -> str:
    return weekly_trend_summary(load_wellness_log(), days=days)


@function_tool
async def sync_goals_to_todoist(ctx: RunContext[Userdata]) -> str:
    entry = ctx.userdata.current_entry
    if not entry or not entry.objectives:
        return "No active check-in or no objectives to sync."
    try:
        session, tools = await get_mcp_client("todoist")
        created = 0
        if isinstance(tools, dict) and "create_task" in tools:
            for o in entry.objectives:
                await session.call_tool("create_task", {"content": o})
                created += 1
        else:
            tool_name = next(iter(tools.keys())) if tools else None
            if tool_name:
                for o in entry.objectives:
                    await session.call_tool(tool_name, {"content": o})
                    created += 1
        return f"Created {created} Todoist task(s)."
    except Exception as e:
        logger.exception("sync_goals_to_todoist failed: %s", e)
        return "Todoist sync failed or not configured."


@function_tool
async def trigger_zapier_tool(ctx: RunContext[Userdata]) -> str:
    entry = ctx.userdata.current_entry
    if not entry:
        return "No active check-in to trigger Zapier for."
    if not ZAPIER_WEBHOOK_URL:
        return "Zapier webhook not configured."
    try:
        import requests
        resp = requests.post(ZAPIER_WEBHOOK_URL, json=entry.to_dict(), timeout=10)
        if 200 <= resp.status_code < 300:
            return "Zapier webhook successfully triggered."
        else:
            return "Zapier webhook returned non-2xx response."
    except Exception as e:
        logger.exception("trigger_zapier_tool failed: %s", e)
        return "Zapier trigger failed."


# -------------------------
# Agent & lifecycle
# -------------------------
AGENT_PROMPT = """
You are CalmCheck, a calm, empathetic Health & Wellness companion.
Ask about mood (optional 1-10), energy (low/medium/high), stress (short), and 1-3 practical objectives.
Offer small, grounded suggestions on request.
Use function tools to record and finalize check-ins.
When user requests, create Todoist tasks or trigger Zapier via webhook.
Reference last check-in at the start of a new session.
"""

class WellnessAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=AGENT_PROMPT,
            tools=[
                start_checkin,
                record_mood,
                record_energy,
                record_stress,
                add_objectives,
                give_suggestion,
                complete_checkin,
                get_wellness_history,
                weekly_summary,
                sync_goals_to_todoist,
                trigger_zapier_tool,
            ],
        )


def prewarm(proc: JobProcess):
    logger.info("Prewarming VAD...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD ready.")


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("Wellness agent starting...")

    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    userdata = Userdata(session_id=session_id)

    logger.info("Wellness log path: %s", wellness_log_path())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage summary: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=WellnessAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    logger.info("Launching Wellness Agent worker...")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
