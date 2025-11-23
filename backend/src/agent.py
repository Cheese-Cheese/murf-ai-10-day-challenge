import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Literal, List, Optional
from dataclasses import dataclass, field, asdict
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
    metrics,
    MetricsCollectedEvent,
    RunContext,
    function_tool,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("wellness-agent")
load_dotenv(".env.local")

# ======================================================
# 🧠 DATA MODELS & STATE
# ======================================================

@dataclass
class CheckInEntry:
    """📝 Schema for a single daily check-in"""
    date: str = field(default_factory=lambda: datetime.now().isoformat())
    mood: str | None = None
    energy_level: str | None = None
    stressors: str | None = None
    objectives: List[str] = field(default_factory=list)
    self_care: str | None = None
    
    def is_complete(self) -> bool:
        """Check if we have enough info to wrap up"""
        # We need at least a mood and one objective
        return self.mood is not None and len(self.objectives) > 0

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class UserSessionData:
    """👤 Session state wrapper"""
    current_entry: CheckInEntry
    history_summary: str = ""  # Context from previous sessions

# ======================================================
# 💾 PERSISTENCE LAYER (Single JSON File)
# ======================================================

DB_FILE = "wellness_log.json"

def get_db_path():
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, DB_FILE)

def load_history_context() -> str:
    """📖 Reads the JSON log and returns a summary of the LAST check-in."""
    path = get_db_path()
    if not os.path.exists(path):
        return "This is the user's first session. Welcome them warmly."

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not data or not isinstance(data, list):
            return "No valid past history found."

        last_entry = data[-1]
        
        # Parse date for friendly display
        try:
            date_obj = datetime.fromisoformat(last_entry.get('date', ''))
            date_str = date_obj.strftime("%A, %B %d")
        except:
            date_str = "the last session"

        summary = (
            f"CONTEXT FROM PREVIOUS SESSION ({date_str}):\n"
            f"- Mood: {last_entry.get('mood', 'unknown')}\n"
            f"- Energy: {last_entry.get('energy_level', 'unknown')}\n"
            f"- Their Goals were: {', '.join(last_entry.get('objectives', []))}\n"
            f"INSTRUCTION: Reference this briefly. E.g., 'Last time you were feeling...'"
        )
        return summary
    except Exception as e:
        logger.error(f"Error reading history: {e}")
        return "Error loading history. Proceed as a fresh session."

def append_entry_to_log(entry: CheckInEntry):
    """💾 Appends the new entry to the JSON list."""
    path = get_db_path()
    data = []

    # Read existing
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    data = json.loads(content)
        except Exception as e:
            logger.error(f"Corrupt DB, starting fresh: {e}")

    # Append new
    data.append(entry.to_dict())

    # Write back
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved check-in to {path}")

# ======================================================
# 🛠️ AGENT TOOLS
# ======================================================

@function_tool
async def log_mood_status(
    ctx: RunContext[UserSessionData],
    mood: Annotated[str, Field(description="The user's reported emotional state (e.g., happy, anxious, calm).")],
    energy: Annotated[str, Field(description="Energy level (e.g., high, low, exhausted, energetic).")],
    stressors: Annotated[str, Field(description="Any specific things stressing them out, or 'None'.")] = "None"
) -> str:
    """📝 Log the user's mood and energy. Call this when they describe how they feel."""
    ctx.userdata.current_entry.mood = mood
    ctx.userdata.current_entry.energy_level = energy
    ctx.userdata.current_entry.stressors = stressors
    
    print(f"🧠 MOOD LOGGED: {mood} | Energy: {energy}")
    
    return f"Logged: Mood is {mood}, energy is {energy}. Now ask about their intentions for the day."

@function_tool
async def log_daily_intentions(
    ctx: RunContext[UserSessionData],
    objectives: Annotated[List[str], Field(description="List of 1-3 practical goals for the day.")],
    self_care: Annotated[str, Field(description="Any specific self-care or rest activity planned.")] = "None"
) -> str:
    """🎯 Log the user's goals/intentions. Call this when they state what they want to do."""
    ctx.userdata.current_entry.objectives = objectives
    ctx.userdata.current_entry.self_care = self_care
    
    print(f"🎯 GOALS LOGGED: {objectives}")
    
    return "Goals logged. Now, offer a brief, grounded reflection or simple advice based on their mood and goals."

@function_tool
async def finalize_checkin(ctx: RunContext[UserSessionData]) -> str:
    """💾 Finalize the session. Call this AFTER doing the recap and confirming with the user."""
    entry = ctx.userdata.current_entry
    
    if not entry.is_complete():
        return "Cannot finalize yet. Please ask for their mood and at least one objective for the day."

    try:
        append_entry_to_log(entry)
        return "Session saved successfully. You can now say goodbye."
    except Exception as e:
        return f"Error saving session: {e}"

# ======================================================
# 🤖 COMPANION AGENT
# ======================================================

class WellnessCompanion(Agent):
    def __init__(self, history_context: str):
        super().__init__(
            instructions=f"""
            You are a supportive, grounded Voice Wellness Companion.
            
            OBJECTIVE:
            Conduct a 2-3 minute daily check-in to track mood and set intentions.
            
            CORE BEHAVIORS:
            1. **Grounded & Warm**: Be kind but practical. Avoid toxic positivity.
            2. **Non-Medical**: NEVER offer medical diagnoses or clinical advice. If the user mentions serious symptoms, suggest they see a professional.
            3. **Brief**: Keep responses short (1-2 sentences) unless explaining an idea.
            
            SESSION FLOW:
            1. **Connect**: Greet them. {history_context}
            2. **Mood Check**: Ask "How are you feeling today?" and "How is your energy?".
            3. **Intentions**: Ask "What are 1-3 things you want to focus on today?" or "Any self-care planned?".
            4. **Reflect**: Offer ONE piece of simple, non-medical advice (e.g., "Since you're tired, maybe break that big goal into 20-minute chunks").
            5. **Recap & Close**: Summarize what they said (Mood + Goals) and ask "Does that sound right?".
            6. **Finalize**: Once they confirm, call the `finalize_checkin` tool.

            TONE:
            Calm, steady, encouraging.
            """,
            tools=[
                log_mood_status,
                log_daily_intentions,
                finalize_checkin
            ],
        )

# ======================================================
# 🚀 MAIN ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # 1. Load History
    history_summary = load_history_context()
    print(f"\n📜 LOADED HISTORY:\n{history_summary}\n")

    # 2. Init Session State
    initial_state = UserSessionData(
        current_entry=CheckInEntry(),
        history_summary=history_summary
    )

    # 3. Setup Agent Session
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", # Calm, conversational voice
            style="Conversation",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=initial_state,
    )

    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    # 4. Start Agent with Dynamic History Context
    await session.start(
        agent=WellnessCompanion(history_context=history_summary),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))