# ======================================================
# 💼 ADVANCED SDR AGENT – SKYFLOW CRM (INDIA)
# 🚀 Features:
# - FAQ Search Tool (company_faq.json)
# - Natural SDR Persona
# - Lead Capture + JSON “CRM notes”
# - Qualification Score + Follow-up Email Draft
# - Murf Falcon TTS + Deepgram STT + Gemini Flash Lite
# ======================================================

import logging
import json
import os
from datetime import datetime
from typing import Annotated, Optional
from dataclasses import dataclass, asdict

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

print("\n" + "💼" * 50)
print("🚀 ADVANCED SDR AGENT LOADED – SKYFLOW CRM")
print("💼" * 50 + "\n")

# ======================================================
# 📂 1. KNOWLEDGE BASE (FAQ)
# ======================================================

FAQ_FILE = "company_faq.json"
LEADS_FILE = "leads_db.json"

def load_faq():
    """
    Load FAQ JSON once at startup.
    Returns list[dict] and also a compact text version for instructions.
    """
    path = os.path.join(os.path.dirname(__file__), FAQ_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{FAQ_FILE} not found. Please create company_faq.json next to agent.py."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Text version for prompt
    text_blocks = []
    for item in data:
        text_blocks.append(f"Q: {item['question']}\nA: {item['answer']}")
    kb_text = "\n\n".join(text_blocks)

    return data, kb_text

FAQ_DATA, FAQ_TEXT = load_faq()

def simple_faq_search(query: str) -> str:
    """
    Tiny keyword-based FAQ matcher.
    Returns best-matching FAQ answer(s) as a text snippet.
    """
    q = query.lower()
    scored = []
    for item in FAQ_DATA:
        text = (item["question"] + " " + item["answer"]).lower()
        score = 0
        for word in q.split():
            if word in text:
                score += 1
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_item = scored[0]

    if best_score == 0:
        return (
            "I couldn’t find an exact match in the FAQ. "
            "I’ll keep it high-level and avoid guessing details."
        )

    # You can also include related questions if you want, but 1 is enough.
    return f"Matched FAQ:\nQ: {best_item['question']}\nA: {best_item['answer']}"

# ======================================================
# 💾 2. LEAD MODEL
# ======================================================

@dataclass
class LeadProfile:
    name: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    use_case: Optional[str] = None
    team_size: Optional[str] = None
    timeline: Optional[str] = None
    budget: Optional[str] = None
    notes: Optional[str] = None   # free-form notes the SDR might infer

    def is_min_qualified(self) -> bool:
        """Basic qualification: must at least have Name + Email + Use Case."""
        return bool(self.name and self.email and self.use_case)

@dataclass
class Userdata:
    lead_profile: LeadProfile

# ======================================================
# 🛠️ 3. TOOLS
# ======================================================

@function_tool
async def faq_lookup(
    ctx: RunContext[Userdata],
    question: Annotated[str, Field(description="Customer's question about product, pricing, features, etc.")]
) -> str:
    """
    🔍 FAQ lookup tool.
    Use this whenever the user asks about product, pricing, features, or company.
    """
    result = simple_faq_search(question)
    print(f"🔍 FAQ LOOKUP for: {question} -> {result[:80]}...")
    return result


@function_tool
async def update_lead_profile(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Customer's name")] = None,
    company: Annotated[Optional[str], Field(description="Customer's company name")] = None,
    email: Annotated[Optional[str], Field(description="Customer's email address")] = None,
    role: Annotated[Optional[str], Field(description="Customer's job title or role")] = None,
    use_case: Annotated[Optional[str], Field(description="What they want to use SkyFlow CRM for")] = None,
    team_size: Annotated[Optional[str], Field(description="Rough team size (e.g. 5, 20, 100+)")] = None,
    timeline: Annotated[Optional[str], Field(description="When they want to start (now / soon / later)")] = None,
    budget: Annotated[Optional[str], Field(description="Any budget indication if they mentioned it")] = None,
    notes: Annotated[Optional[str], Field(description="Any extra qualification notes")] = None,
) -> str:
    """
    ✍️ Captures / updates lead fields during the conversation.
    Call this only when the user has clearly provided that info.
    """
    profile = ctx.userdata.lead_profile

    if name: profile.name = name
    if company: profile.company = company
    if email: profile.email = email
    if role: profile.role = role
    if use_case: profile.use_case = use_case
    if team_size: profile.team_size = team_size
    if timeline: profile.timeline = timeline
    if budget: profile.budget = budget
    if notes: profile.notes = notes

    print(f"📝 LEAD UPDATED: {profile}")
    return "Lead profile updated. Acknowledge gently and continue the conversation."


@function_tool
async def finalize_lead_and_summarize(
    ctx: RunContext[Userdata],
) -> str:
    """
    💾 Saves the lead to JSON + generates:
    - Verbal summary hint
    - CRM-style notes + fit score
    - Follow-up email suggestion
    """
    profile = ctx.userdata.lead_profile
    entry = asdict(profile)
    entry["timestamp"] = datetime.now().isoformat()

    # --- Use LLM to generate CRM notes + fit_score + email ---
    llm = ctx.userdata.agent_session.llm  # type: ignore[attr-defined]

    prompt = f"""
You are a sales assistant creating CRM notes based on this structured lead data:

{json.dumps(entry, indent=2)}

Return a JSON object with:
- "summary": 1-2 sentence description of who they are and what they want.
- "pain_points": list of 1-3 short pain points or goals, if you can infer from use_case; else empty list.
- "decision_role": "decision_maker" | "influencer" | "unknown" based on role/company.
- "urgency": "now" | "soon" | "later" | "unknown" based on timeline.
- "fit_score": integer 0-100 (higher = better fit for SkyFlow CRM).
- "email_subject": short subject line for a follow-up email.
- "email_body": a short 2-3 paragraph follow-up email body, friendly and concise.

Only reply with valid JSON.
"""

    try:
        llm_raw = await llm.aask(prompt)
        extra = json.loads(llm_raw)
    except Exception as e:
        print(f"⚠️ LLM notes generation failed: {e}")
        extra = {
            "summary": "Potential lead for SkyFlow CRM.",
            "pain_points": [],
            "decision_role": "unknown",
            "urgency": "unknown",
            "fit_score": 50,
            "email_subject": "Thanks for chatting with SkyFlow CRM",
            "email_body": "Hi there,\n\nThanks for your time today. We will follow up with more details.\n\nBest,\nSkyFlow CRM SDR"
        }

    entry["crm_summary"] = extra.get("summary")
    entry["pain_points"] = extra.get("pain_points")
    entry["decision_role"] = extra.get("decision_role")
    entry["urgency"] = extra.get("urgency")
    entry["fit_score"] = extra.get("fit_score")
    entry["email_subject"] = extra.get("email_subject")
    entry["email_body"] = extra.get("email_body")

    # --- Save to JSON DB ---
    db_path = os.path.join(os.path.dirname(__file__), LEADS_FILE)
    existing_data = []
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = []

    existing_data.append(entry)

    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4)

    print(f"✅ LEAD + NOTES SAVED TO {LEADS_FILE}")

    # This string is a hint to the agent on what to say to the user.
    name = profile.name or "there"
    use_case = profile.use_case or "your use case"
    email = profile.email or "your email"

    return f"""
    Lead saved. Here is your response prompt:

    Thank you {name}, I've noted that you are interested in {use_case}. 
    We will follow up at {email} with next steps. 
    Have a great day!

    Now politely end the call.
    """



# ======================================================
# 🤖 4. SDR AGENT
# ======================================================

class SkyflowSDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
You are **Riya**, a friendly and professional Sales Development Representative (SDR)
for **SkyFlow CRM**, a sales CRM and outreach automation platform for Indian startups.

You have three main jobs:
1) Welcome the visitor and understand what they're working on.
2) Answer questions about SkyFlow CRM ONLY using the FAQ + faq_lookup tool.
3) Collect lead details and close the conversation with a clear summary.

📘 FAQ KNOWLEDGE (for context):
{FAQ_TEXT}

🧠 HOW TO BEHAVE:

1. Greeting & Discovery
   - Start with a warm, concise greeting.
   - Ask what brought them here and what they are working on.
   - Keep the tone conversational, not robotic.

2. Answering Questions (FAQ)
   - Whenever they ask about product, pricing, features, or who it's for:
     → Call faq_lookup(question=...) to fetch relevant FAQ text.
     → Then answer in your own words based on that tool result.
   - If you truly cannot answer from FAQ, say:
     "I’ll have someone from the team email you detailed info."

3. Lead Capture (VERY IMPORTANT)
   Over the course of the conversation, gently collect:
   - Name
   - Company
   - Role
   - Email
   - Use case (what they want SkyFlow CRM for)
   - Team size
   - Timeline (now / soon / later)
   - Budget (optional, if it comes up)

   When the user provides any of these, call update_lead_profile()
   immediately with just the fields they gave.

   Example pattern:
   - Answer their question.
   - Then add a soft qualifier:
     "By the way, what does your team currently use to manage leads?"

4. Ending the Call
   - When they say things like "that's all", "I'm done", "thanks", or clearly
     indicate they are finished:
     → If you have at least name + email + use_case, call finalize_lead_and_summarize().
     → If you are missing key info, try to politely ask once. If they still
       want to end, call finalize_lead_and_summarize() with whatever you have.
   - After finalize_lead_and_summarize() returns, follow its instructions and say goodbye.

5. Style
   - Keep answers clear and concise.
   - Sound like a human SDR, not a chatbot.
   - NEVER invent specific prices, plan names, or technical claims that are
     not in the FAQ text or faq_lookup result.

Do NOT mention that you are an AI, an assignment, or part of any challenge.
Just act like a normal SDR for SkyFlow CRM.
""",
            tools=[faq_lookup, update_lead_profile, finalize_lead_and_summarize],
        )

# ======================================================
# 🎬 5. ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "💼" * 25)
    print("🚀 STARTING SKYFLOW CRM SDR SESSION")
    print("💼" * 25)

    userdata = Userdata(lead_profile=LeadProfile())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.0-flash-lite", temperature=0.3, max_output_tokens=256),
        tts=murf.TTS(
            voice="en-US-natalie",   # nice professional Murf Falcon voice
            style="Promo",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    # so tools can access llm in finalize_lead_and_summarize
    userdata.agent_session = session  # type: ignore[attr-defined]

    await session.start(
        agent=SkyflowSDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
