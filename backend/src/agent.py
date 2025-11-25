import logging
import json
import os
import asyncio
from typing import Annotated, Literal, Optional, Dict, List
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
    function_tool,
    RunContext,
)

# 🔌 PLUGINS
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 📚 KNOWLEDGE BASE (PHYSICS DATA)
# ======================================================

CONTENT_FILE = "physics_content.json" 

DEFAULT_CONTENT = [
  {
    "id": "newton_laws",
    "title": "Newton's Laws of Motion",
    "summary": "Sir Isaac Newton's three laws of motion describe the relationship between the motion of an object and the forces acting on it. The first law is inertia, the second is F=ma, and the third states that for every action, there is an equal and opposite reaction.",
    "sample_question": "Explain Newton's Third Law of Motion and give a real-world example."
  },
  {
    "id": "energy",
    "title": "Energy",
    "summary": "Energy is the quantitative property that must be transferred to a body or physical system to perform work on the body, or to heat it. Common forms include Kinetic Energy (motion) and Potential Energy (position). Energy cannot be created or destroyed, only transformed.",
    "sample_question": "What is the difference between Kinetic Energy and Potential Energy?"
  },
  {
    "id": "gravity",
    "title": "Gravity",
    "summary": "Gravity is a fundamental interaction which causes mutual attraction between all things with mass or energy. On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides.",
    "sample_question": "How does mass affect the gravitational force between two objects according to Newton's law of universal gravitation?"
  },
  {
    "id": "thermodynamics",
    "title": "Thermodynamics",
    "summary": "Thermodynamics is the branch of physics that deals with heat, work, and temperature, and their relation to energy, entropy, and the physical properties of matter and radiation.",
    "sample_question": "What does the Second Law of Thermodynamics say about entropy in an isolated system?"
  }
]

def load_content():
    try:
        path = os.path.join(os.path.dirname(__file__), CONTENT_FILE)
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_CONTENT, f, indent=4)
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        logger.error(f"Error loading content: {e}")
        return []

COURSE_CONTENT = load_content()

# ======================================================
# 🧠 STATE MANAGEMENT (With Richer Mastery)
# ======================================================

@dataclass
class TopicMastery:
    times_explained: int = 0
    times_quizzed: int = 0
    times_taught_back: int = 0
    last_score: int = 0
    avg_score: float = 0.0
    _score_history: List[int] = field(default_factory=list)

    def add_score(self, score: int):
        self.last_score = score
        self.times_taught_back += 1
        self._score_history.append(score)
        self.avg_score = sum(self._score_history) / len(self._score_history)

@dataclass
class TutorState:
    """🧠 Tracks the current learning context and mastery scores"""
    current_topic_id: str | None = None
    current_topic_data: dict | None = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"
    mastery: Dict[str, TopicMastery] = field(default_factory=dict)
    
    def set_topic(self, topic_id: str):
        topic = next((item for item in COURSE_CONTENT if item["id"] == topic_id), None)
        if topic:
            self.current_topic_id = topic_id
            self.current_topic_data = topic
            # Initialize mastery entry if not exists
            if topic_id not in self.mastery:
                self.mastery[topic_id] = TopicMastery()
            return True
        return False
        
    def get_mastery(self, topic_id: str) -> TopicMastery:
        if topic_id not in self.mastery:
            self.mastery[topic_id] = TopicMastery()
        return self.mastery[topic_id]

@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None 

# ======================================================
# 🛠️ TUTOR TOOLS
# ======================================================

@function_tool
async def select_topic(
    ctx: RunContext[Userdata], 
    topic_id: Annotated[str, Field(description="The ID of the topic to study")]
) -> str:
    """📚 Selects a physics topic to study from the available list."""
    state = ctx.userdata.tutor_state
    success = state.set_topic(topic_id.lower())
    
    if success:
        m = state.get_mastery(topic_id)
        stats = f"(Mastery: {m.avg_score:.1f}% | Taught back: {m.times_taught_back} times)"
        return f"Topic set to {state.current_topic_data['title']} {stats}. Ask the user if they want to 'Learn', be 'Quizzed', or 'Teach it back'."
    else:
        available = ", ".join([t["id"] for t in COURSE_CONTENT])
        return f"Topic not found. Available topics are: {available}"

@function_tool
async def set_learning_mode(
    ctx: RunContext[Userdata], 
    mode: Annotated[str, Field(description="The mode to switch to: 'learn', 'quiz', or 'teach_back'")]
) -> str:
    """🔄 Switches the interaction mode, updates voice, and increments usage counters."""
    state = ctx.userdata.tutor_state
    
    if not state.current_topic_id:
        return "Please select a topic first using select_topic."

    state.mode = mode.lower()
    mastery = state.get_mastery(state.current_topic_id)
    
    agent_session = ctx.userdata.agent_session 
    
    if agent_session:
        if state.mode == "learn":
            mastery.times_explained += 1
            agent_session.tts.update_options(voice="en-US-matthew", style="Promo")
            instruction = f"Mode: LEARN. Explain this: {state.current_topic_data['summary']}"
            
        elif state.mode == "quiz":
            mastery.times_quizzed += 1
            agent_session.tts.update_options(voice="en-US-alicia", style="Conversational")
            instruction = f"Mode: QUIZ. Ask this: {state.current_topic_data['sample_question']}"
            
        elif state.mode == "teach_back":
            # Don't increment count yet, wait for evaluation
            agent_session.tts.update_options(voice="en-US-ken", style="Promo")
            instruction = "Mode: TEACH_BACK. Ask the user to explain the concept to you."
        else:
            return "Invalid mode."
    else:
        instruction = "Voice switch failed."

    print(f"🔄 MODE -> {state.mode.upper()} | Stats for {state.current_topic_id}: {mastery}")
    return f"Switched to {state.mode} mode. {instruction}"

@function_tool
async def evaluate_teaching(
    ctx: RunContext[Userdata],
    user_explanation: Annotated[str, Field(description="The explanation given by the user")],
    score: Annotated[int, Field(description="A score between 0-100 based on accuracy and clarity")]
) -> str:
    """📝 Records the teach-back score and returns feedback instructions."""
    state = ctx.userdata.tutor_state
    
    if not state.current_topic_id:
        return "No topic selected."

    # Update Mastery
    mastery = state.get_mastery(state.current_topic_id)
    mastery.add_score(score)
    
    print(f"📝 EVALUATION: Score {score}/100 | Avg {mastery.avg_score:.1f} | Explanation: {user_explanation[:50]}...")
    
    return (
        f"User Score: {score}/100. Running Average: {mastery.avg_score:.1f}. "
        f"Give specific feedback on their explanation. "
        f"If score < 70, correct their mistakes gently. If > 90, praise their mastery."
    )

# ======================================================
# 🧠 AGENT DEFINITION
# ======================================================

class TutorAgent(Agent):
    def __init__(self):
        topic_list = ", ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])
        
        super().__init__(
            instructions=f"""
            You are a Physics Tutor helping users master concepts like Newton's Laws.
            
            📚 **AVAILABLE TOPICS:** {topic_list}
            
            🔄 **MODES:**
            1. **LEARN (Matthew):** Explain the concept.
            2. **QUIZ (Alicia):** Ask a question.
            3. **TEACH_BACK (Ken):** Listen to the user's explanation.
            
            ⚙️ **RULES:**
            - **Always** select a topic first.
            - **Always** use `set_learning_mode` to switch tasks.
            - **In Teach-Back Mode:** Listen to the user, **decide on a score (0-100)** based on their accuracy, and call `evaluate_teaching` with that score.
            """,
            tools=[select_topic, set_learning_mode, evaluate_teaching],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "⚛️" * 25)
    print("🚀 STARTING PHYSICS TUTOR SESSION (WITH MASTERY TRACKING)")
    
    userdata = Userdata(tutor_state=TutorState())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew", style="Promo", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )
    
    userdata.agent_session = session
    
    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))