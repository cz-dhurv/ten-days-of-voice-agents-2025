# ======================================================
# 🚀 ADVANCED ACTIVE RECALL TUTOR (WITH SQLITE MEMORY)
# 🎯 Features:
# - Learn / Quiz / Teach-Back Modes
# - SQLite Mastery Tracking
# - Teach-Back Scoring (0–100)
# - Weakest-Concept Lookup
# - Mode-Based Voice Switching (Murf Falcon)
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
# 📚 LOAD COURSE CONTENT
# ======================================================

CONTENT_FILE = "shared-data/day4_tutor_content.json"

def load_content():
    path = os.path.join(os.path.dirname(__file__), CONTENT_FILE)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

COURSE_CONTENT = load_content()

ALL_TOPICS = {c["id"]: c for c in COURSE_CONTENT}

# ======================================================
# 🗄️ SQLITE DATABASE SETUP
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

def update_mastery(topic_id: str, field: str, score: Optional[float] = None):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # increment field counter
    cur.execute(f"UPDATE mastery SET {field} = {field} + 1 WHERE topic_id=?", (topic_id,))

    if score is not None:
        # Update last_score
        cur.execute("UPDATE mastery SET last_score=? WHERE topic_id=?", (score, topic_id))

        # Update avg_score
        cur.execute("""
            UPDATE mastery
            SET avg_score = (avg_score + ?) / 2
            WHERE topic_id=?
        """, (score, topic_id))

    conn.commit()
    conn.close()

def get_weakest_topics():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        SELECT topic_id, avg_score FROM mastery
        ORDER BY avg_score ASC LIMIT 3
    """)
    data = cur.fetchall()
    conn.close()
    return data

# Initialize DB on import
init_db()

# ======================================================
# 🎯 STATE
# ======================================================

@dataclass
class TutorState:
    current_topic_id: str | None = None
    current_topic: dict | None = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"

    def set_topic(self, topic_id: str):
        topic = ALL_TOPICS.get(topic_id)
        if not topic:
            return False
        self.current_topic_id = topic_id
        self.current_topic = topic
        ensure_topic_in_db(topic_id)
        return True


@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None

# ======================================================
# 🛠️ TOOLS
# ======================================================

@function_tool
async def select_topic(
    ctx: RunContext[Userdata],
    topic_id: Annotated[str, Field(description="Topic ID to study")]
):
    state = ctx.userdata.tutor_state
    topic_id = topic_id.lower()

    if not state.set_topic(topic_id):
        return f"Topic not found. Available: {', '.join(ALL_TOPICS.keys())}"

    return f"Topic set to {state.current_topic['title']}. Ask the user which mode to continue with."


@function_tool
async def set_mode(
    ctx: RunContext[Userdata],
    mode: Annotated[str, Field(description="Mode: learn, quiz, teach_back")]
):
    mode = mode.lower()
    state = ctx.userdata.tutor_state
    session = ctx.userdata.agent_session

    if mode not in ["learn", "quiz", "teach_back"]:
        return "Invalid mode. Available: learn, quiz, teach_back."

    state.mode = mode

    # Voice switching
    if session:
        if mode == "learn":
            session.tts.update_options(voice="en-US-matthew")
            update_mastery(state.current_topic_id, "times_explained")
            return f"Explain: {state.current_topic['summary']}"

        elif mode == "quiz":
            session.tts.update_options(voice="en-US-alicia")
            update_mastery(state.current_topic_id, "times_quizzed")
            return f"Question: {state.current_topic['sample_question']}"

        elif mode == "teach_back":
            session.tts.update_options(voice="en-US-ken")
            return "Explain this concept in your own words."

    return "Mode updated."


@function_tool
async def evaluate_teach_back(
    ctx: RunContext[Userdata],
    explanation: Annotated[str, Field(description="User's explanation")]
):
    """
    LLM evaluates explanation (0–100), stores score in SQLite, updates mastery.
    """
    state = ctx.userdata.tutor_state

    evaluation_prompt = f"""
    Score the user's explanation of the following concept: {state.current_topic['title']}

    Concept Summary:
    {state.current_topic['summary']}

    User explanation:
    {explanation}

    Respond in JSON:
    {{
        "score": number 0-100,
        "feedback": "1-2 sentence feedback"
    }}
    """

    llm = ctx.userdata.agent_session.llm
    result = await llm.aask(evaluation_prompt)

    try:
        parsed = json.loads(result)
        score = parsed["score"]
        feedback = parsed["feedback"]
    except:
        score = 50
        feedback = "Could not parse the explanation properly."

    update_mastery(state.current_topic_id, "times_taught_back", score=score)

    return f"Your score: {score}/100.\nFeedback: {feedback}"


@function_tool
async def weakest_concepts(
    ctx: RunContext[Userdata]
):
    data = get_weakest_topics()

    if not data:
        return "No mastery data available yet."

    out = []
    for topic_id, score in data:
        out.append(f"{topic_id}: avg_score={round(score,1)}")

    return "Your weakest concepts:\n" + "\n".join(out)

# ======================================================
# 🤖 ADVANCED TUTOR AGENT
# ======================================================

class AdvancedTutor(Agent):
    def __init__(self):
        topics = ", ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])

        super().__init__(
            instructions=f"""
You are an advanced learning tutor that uses active recall, concept explanation,
teaching by example, quizzing, and teach-back evaluation.

Capabilities:
• Multi-mode operation (learn, quiz, teach-back)
• Teach-back evaluation (0–100)
• Tracks user mastery using SQLite
• Can identify user's weakest concepts
• Switches voice based on mode (Murf voices)

Workflow:
1. First ask: "Which concept would you like to learn today?"
2. If user mentions a topic → call select_topic()
3. If user says learn/quiz/teach → call set_mode()
4. In teach_back mode, after user explanation → call evaluate_teach_back()
5. If user asks "Which concepts am I weak in?" → call weakest_concepts()

Do not mention that you are part of any challenge. 
Do not say this is Day 4. 
Be natural, professional, engaging.
""",
            tools=[select_topic, set_mode, evaluate_teach_back, weakest_concepts]
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    userdata = Userdata(tutor_state=TutorState())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    userdata.agent_session = session

    await session.start(
        agent=AdvancedTutor(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        )
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
