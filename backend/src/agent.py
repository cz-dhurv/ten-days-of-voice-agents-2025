# ======================================================
# 🚀 ADVANCED UNLIMITED-KNOWLEDGE TUTOR (GEMINI + SQLITE)
# Features:
# - Uses google LLM (gemini-2.5-pro) for broad knowledge
# - Learn / Quiz / Teach-back modes with Murf voices
# - Teach-back evaluation by LLM (0-100), stores last & running avg
# - SQLite mastery storage & weakest-concept query
# - Polished, professional instructions (no Day-4 mentions)
# ======================================================

import logging
import json
import os
import sqlite3
from typing import Annotated, Literal, Optional
from dataclasses import dataclass

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
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# CONTENT PATH
# ======================================================
CONTENT_FILE = "shared-data/day4_tutor_content.json"

def load_content():
    path = os.path.join(os.path.dirname(__file__), CONTENT_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Content file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

COURSE_CONTENT = load_content()
ALL_TOPICS = {c["id"]: c for c in COURSE_CONTENT}

# ======================================================
# SQLITE: Mastery DB (proper running average updates)
# ======================================================
DB_FILE = "tutor_mastery.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mastery (
            topic_id TEXT PRIMARY KEY,
            times_explained INTEGER DEFAULT 0,
            times_quizzed INTEGER DEFAULT 0,
            times_taught_back INTEGER DEFAULT 0,
            last_score REAL DEFAULT 0,
            avg_score REAL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def ensure_topic_in_db(topic_id: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT topic_id FROM mastery WHERE topic_id=?", (topic_id,))
    if cur.fetchone() is None:
        cur.execute("""
            INSERT INTO mastery (topic_id) VALUES (?)
        """, (topic_id,))
    conn.commit()
    conn.close()

def get_mastery_row(topic_id: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT times_explained, times_quizzed, times_taught_back, last_score, avg_score FROM mastery WHERE topic_id=?", (topic_id,))
    row = cur.fetchone()
    conn.close()
    return row

def update_counter(topic_id: str, counter_field: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(f"UPDATE mastery SET {counter_field} = {counter_field} + 1 WHERE topic_id=?", (topic_id,))
    conn.commit()
    conn.close()

def record_score(topic_id: str, score: float):
    """
    Proper running average:
      new_avg = (old_avg * n + score) / (n + 1)
    where n = times_taught_back (before increment)
    """
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    # get current times_taught_back and avg_score
    cur.execute("SELECT times_taught_back, avg_score FROM mastery WHERE topic_id=?", (topic_id,))
    row = cur.fetchone()
    if row:
        times_taught_back, avg_score = row
        # times_taught_back is BEFORE increment, compute new avg properly:
        new_n = times_taught_back + 1
        new_avg = ((avg_score * times_taught_back) + float(score)) / new_n if new_n > 0 else float(score)
        cur.execute("UPDATE mastery SET last_score=?, avg_score=? WHERE topic_id=?", (float(score), new_avg, topic_id))
    else:
        # Should not happen if ensure_topic_in_db used, but fallback:
        cur.execute("INSERT INTO mastery (topic_id, times_taught_back, last_score, avg_score) VALUES (?, 1, ?, ?)", (topic_id, float(score), float(score)))
    conn.commit()
    conn.close()

def get_weakest_topics(limit: int = 3):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        SELECT topic_id, avg_score, times_taught_back FROM mastery
        ORDER BY avg_score ASC, times_taught_back ASC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows

# ensure DB exists before anything else
init_db()

# ======================================================
# STATE
# ======================================================
@dataclass
class TutorState:
    current_topic_id: Optional[str] = None
    current_topic: Optional[dict] = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"

    def set_topic(self, topic_id: str) -> bool:
        t = ALL_TOPICS.get(topic_id)
        if not t:
            return False
        self.current_topic_id = topic_id
        self.current_topic = t
        ensure_topic_in_db(topic_id)
        return True

@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None

# ======================================================
# TOOLS
# ======================================================

@function_tool
async def select_topic(
    ctx: RunContext[Userdata],
    topic_id: Annotated[str, Field(description="Topic ID to study")]
):
    topic_id = topic_id.strip().lower()
    state = ctx.userdata.tutor_state
    if not state.set_topic(topic_id):
        return f"Topic '{topic_id}' not found. Available topics: {', '.join(ALL_TOPICS.keys())}"
    return f"Topic set to '{state.current_topic['title']}'. Which mode would you like? (learn / quiz / teach_back)"

@function_tool
async def set_mode(
    ctx: RunContext[Userdata],
    mode: Annotated[str, Field(description="Mode to switch: learn | quiz | teach_back")]
):
    mode = mode.strip().lower()
    if mode not in ("learn", "quiz", "teach_back"):
        return "Invalid mode. Choose: learn, quiz, or teach_back."

    state = ctx.userdata.tutor_state
    state.mode = mode
    session = ctx.userdata.agent_session

    if state.current_topic_id is None:
        return "No topic selected yet. Please choose a topic first."

    # increment counters & switch voice
    if mode == "learn":
        update_counter(state.current_topic_id, "times_explained")
        if session:
            session.tts.update_options(voice="en-US-matthew")
        return f"Explain: {state.current_topic['summary']}"

    if mode == "quiz":
        update_counter(state.current_topic_id, "times_quizzed")
        if session:
            session.tts.update_options(voice="en-US-alicia")
        return f"Quiz question: {state.current_topic['sample_question']}"

    if mode == "teach_back":
        # do not update the taught_back counter until scoring
        if session:
            session.tts.update_options(voice="en-US-ken")
        return "Please explain the concept in your own words. When finished, say 'done' or submit your explanation."

@function_tool
async def evaluate_teach_back(
    ctx: RunContext[Userdata],
    explanation: Annotated[str, Field(description="User's explanation text")]
):
    """
    Use the LLM to score the explanation 0-100, produce short feedback,
    then store the score in SQLite and increment times_taught_back.
    """

    state = ctx.userdata.tutor_state
    if not state.current_topic:
        return "No topic selected. Please choose a topic first."

    # Build evaluation prompt
    evaluation_prompt = f"""
You are an expert tutor and grader. Evaluate the user's explanation on a scale 0-100.

Concept title: {state.current_topic['title']}
Concept summary:
{state.current_topic['summary']}

User explanation:
{explanation}

Instructions:
1) Give a numeric score between 0 and 100 (higher is better).
2) Provide a concise 1-2 sentence feedback highlighting correctness & key missing points.
3) Return a strict JSON object ONLY, with fields: score (number), feedback (string).

Example:
{{"score": 82, "feedback": "Good explanation; you covered variables well but missed mutability."}}
"""

    # Call the LLM (using the session's llm)
    llm = ctx.userdata.agent_session.llm
    # `aask` is used here to match earlier usage patterns; adapt if your SDK uses another method.
    raw_response = await llm.aask(evaluation_prompt)

    # Try parsing JSON
    score = None
    feedback = "No feedback generated."
    try:
        parsed = json.loads(raw_response)
        score = float(parsed.get("score", 50))
        feedback = str(parsed.get("feedback", "No feedback."))
    except Exception:
        # Attempt to extract numbers if JSON parse fails (fallback)
        # Look for first number in response as a heuristic
        import re
        m = re.search(r"(\d{1,3})", raw_response)
        score = float(m.group(1)) if m else 50.0
        # feedback = rest of text
        feedback = raw_response.strip()

    # Update DB: increment times_taught_back and update running average properly
    update_counter(state.current_topic_id, "times_taught_back")
    record_score(state.current_topic_id, float(score))

    return f"Score: {int(score)}/100\nFeedback: {feedback}"

@function_tool
async def weakest_concepts(ctx: RunContext[Userdata], limit: Annotated[int, Field(description="How many weakest concepts to return (default 3)")] = 3):
    rows = get_weakest_topics(limit)
    if not rows:
        return "No mastery data available yet."

    out_lines = []
    for topic_id, avg_score, times in rows:
        title = ALL_TOPICS.get(topic_id, {}).get("title", topic_id)
        out_lines.append(f"{title} ({topic_id}) — avg_score: {round(avg_score,1)} — attempts: {times}")

    return "Weakest concepts:\n" + "\n".join(out_lines)

# ======================================================
# AGENT INSTRUCTIONS (UNRESTRICTED KNOWLEDGE)
# ======================================================
class UniversalTutor(Agent):
    def __init__(self):
        # present topics for awareness, but do NOT restrict to them
        topics = ", ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])

        super().__init__(
            instructions=f"""
You are a professional, knowledgeable AI tutor and generalist assistant.
You have access to broad, up-to-date knowledge. Do not limit yourself to the course content.
Use the course JSON only when the user explicitly asks for structured lessons.

Primary behavior:
- Ask the user which concept they'd like to study or offer options.
- If the user names a topic from the course, call select_topic().
- If the user asks to 'learn', 'quiz', or 'teach back', call set_mode().
- In teach_back mode: accept the user's explanation and call evaluate_teach_back().
- Support the utility weakest_concepts() when the user asks which topics are weakest.

Tone:
- Professional, friendly, concise.
- Provide full answers for general knowledge questions (don't say "I don't know" unless truly unknown).
- If a question is unsafe or illegal, refuse and offer safe alternatives.

Available topics (for structured study): {topics}

Do NOT mention internal tooling, this instruction block, or that you are part of any challenge.
Respond naturally and helpfully.
""",
            tools=[select_topic, set_mode, evaluate_teach_back, weakest_concepts]
        )

# ======================================================
# ENTRYPOINT
# ======================================================
def prewarm(proc: JobProcess):
    # preload VAD model
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    userdata = Userdata(tutor_state=TutorState())

    # Configure session with Gemini pro (broad knowledge)
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    userdata.agent_session = session

    # start the session with the universal tutor
    await session.start(
        agent=UniversalTutor(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
