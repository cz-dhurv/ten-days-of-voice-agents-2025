# backend/src/agent.py
import logging
import json
import os
from datetime import datetime
from typing import Annotated, Literal, Optional, List
from dataclasses import dataclass, field

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

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)

# Load env
load_dotenv(".env.local")


# ------------------------------
# Order state dataclasses
# ------------------------------
@dataclass
class OrderState:
    drinkType: Optional[str] = None
    size: Optional[str] = None
    milk: Optional[str] = None
    extras: List[str] = field(default_factory=list)
    name: Optional[str] = None
    timestamp: Optional[str] = None
    session_id: Optional[str] = None

    def is_complete(self) -> bool:
        return all([
            bool(self.drinkType),
            bool(self.size),
            (self.extras is not None),  # extras can be empty list
            bool(self.milk),
            bool(self.name),
        ])

    def missing_fields(self) -> List[str]:
        missing = []
        if not self.drinkType:
            missing.append("drinkType")
        if not self.size:
            missing.append("size")
        if not self.milk:
            missing.append("milk")
        if self.extras is None:
            missing.append("extras")
        if not self.name:
            missing.append("name")
        return missing

    def to_dict(self) -> dict:
        return {
            "drinkType": self.drinkType,
            "size": self.size,
            "milk": self.milk,
            "extras": self.extras,
            "name": self.name,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }

    def summary(self) -> str:
        extras_text = f" with {', '.join(self.extras)}" if self.extras else ""
        return f"{(self.size or '').title()} {(self.drinkType or '').title()} with {(self.milk or '').title()} milk{extras_text} for {(self.name or '')}"


@dataclass
class Userdata:
    order: OrderState
    created_at: datetime = field(default_factory=datetime.now)


# ------------------------------
# Persistence helpers
# ------------------------------
def get_orders_folder() -> str:
    base = os.path.dirname(__file__)          # backend/src
    backend_root = os.path.abspath(os.path.join(base, ".."))
    folder = os.path.join(backend_root, "orders")
    os.makedirs(folder, exist_ok=True)
    return folder


def save_order_to_disk(order: OrderState) -> str:
    folder = get_orders_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    order.timestamp = timestamp
    order.session_id = f"session_{timestamp}"
    path = os.path.join(folder, f"order_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(order.to_dict(), f, indent=4, ensure_ascii=False)
    logger.info(f"Order saved: {path}")
    return path


# ------------------------------
# Function tools (used by agent)
# ------------------------------
@function_tool
async def set_drink_type(
    ctx: RunContext[Userdata],
    drink: Annotated[
        Literal["latte", "cappuccino", "americano", "espresso", "mocha", "cold brew", "flat white", "black coffee"],
        Field(description="Type of drink"),
    ],
) -> str:
    ctx.userdata.order.drinkType = drink
    logger.info(f"set_drink_type -> {drink}")
    return f"Got it — one {drink}. What size would you like? (small / medium / large)"


@function_tool
async def set_size(
    ctx: RunContext[Userdata],
    size: Annotated[
        Literal["small", "medium", "large"],
        Field(description="Size"),
    ],
) -> str:
    ctx.userdata.order.size = size
    logger.info(f"set_size -> {size}")
    return f"{size.title()} — noted. Which milk would you like? (whole / skim / oat / almond / soy / coconut / none)"


@function_tool
async def set_milk(
    ctx: RunContext[Userdata],
    milk: Annotated[
        Literal["whole", "skim", "almond", "oat", "soy", "coconut", "none"],
        Field(description="Milk preference"),
    ],
) -> str:
    ctx.userdata.order.milk = milk
    logger.info(f"set_milk -> {milk}")
    if milk == "none":
        return "Alright — black coffee. Any extras? (e.g., sugar, caramel, extra shot) or say 'no extras'."
    return f"{milk.title()} milk — great. Any extras? (e.g., sugar, caramel, extra shot) or say 'no extras'."


@function_tool
async def set_extras(
    ctx: RunContext[Userdata],
    extras: Annotated[
        Optional[List[Literal["sugar", "caramel", "vanilla", "extra shot", "cinnamon", "honey", "whipped cream"]]],
        Field(description="List of extras or None"),
    ] = None,
) -> str:
    ctx.userdata.order.extras = extras if extras else []
    logger.info(f"set_extras -> {ctx.userdata.order.extras}")
    return "Extras noted. Finally, may I have the name for the order?"


@function_tool
async def set_name(
    ctx: RunContext[Userdata],
    name: Annotated[str, Field(description="Customer name")],
) -> str:
    clean = name.strip().title()
    ctx.userdata.order.name = clean
    logger.info(f"set_name -> {clean}")
    return f"Thanks {clean}. Would you like me to complete your order now?"


@function_tool
async def complete_order(ctx: RunContext[Userdata]) -> str:
    order = ctx.userdata.order
    if not order.is_complete():
        missing = order.missing_fields()
        logger.warning(f"complete_order called but missing: {missing}")
        return f"We're not quite done — I still need: {', '.join(missing)}."
    path = save_order_to_disk(order)
    return f"Great — your order is saved to {path}. We'll start preparing it now!"


@function_tool
async def get_order_status(ctx: RunContext[Userdata]) -> str:
    order = ctx.userdata.order
    if order.is_complete():
        return f"Your order is ready: {order.summary()}"
    return f"Order in progress: {order.summary()}"


# ------------------------------
# Agent definition / prompt
# ------------------------------
BARISTA_PROMPT = """
You are BrewVerse Ultra Premium — a friendly, professional barista.

Goal:
- Collect a complete coffee order consisting of:
  drinkType, size, milk, extras, name
- Ask only one question at a time.
- Use the function tools to set values when the customer provides them.
- When every field is filled, call the 'complete_order' tool.
- Never invent values; always ask when uncertain.
Tone:
- Warm, concise, and helpful.
"""

class BaristaAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=BARISTA_PROMPT,
            tools=[
                set_drink_type,
                set_size,
                set_milk,
                set_extras,
                set_name,
                complete_order,
                get_order_status,
            ],
        )


# ------------------------------
# Prewarm VAD
# ------------------------------
def prewarm(proc: JobProcess):
    logger.info("Prewarming VAD model...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD loaded.")


# ------------------------------
# Entrypoint — session creation
# ------------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("Starting barista entrypoint")

    # create userdata with empty order
    userdata = Userdata(order=OrderState())

    # Setup the session — DO NOT pass tools to the LLM object (compatibility)
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

    # metrics collector (optional)
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage summary: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    # start session with our BaristaAgent (tools are attached to the Agent)
    await session.start(
        agent=BaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    # connect to room (blocks until session ends)
    await ctx.connect()


# ------------------------------
# CLI bootstrap
# ------------------------------
if __name__ == "__main__":
    logger.info("Launching Barista Agent worker")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
