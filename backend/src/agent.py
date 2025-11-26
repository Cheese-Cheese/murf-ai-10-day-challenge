import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Literal, Optional, List
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

# 🔌 PLUGINS
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 📂 1. KNOWLEDGE BASE (FAQ) - LENSKART THEMED
# ======================================================

FAQ_FILE = "lenskart_faq.json"
LEADS_FILE = "lenskart_leads.json"
EMAILS_FILE = "email_drafts.json"

# Default FAQ data for "Lenskart"
DEFAULT_FAQ = [
    {
        "question": "What products do you sell?",
        "answer": "We offer a wide range of eyewear including premium eyeglasses, computer glasses (Blu-cut), polarized sunglasses, and contact lenses. We feature brands like Vincent Chase, John Jacobs, and Lenskart Air."
    },
    {
        "question": "Do you offer home eye check-ups?",
        "answer": "Yes! We offer a 'Home Eye Check-up' service. A certified optometrist will visit your home with specialized equipment and 100 best-selling frames for you to try. It costs just ₹99."
    },
    {
        "question": "What is the Gold Membership?",
        "answer": "Lenskart Gold Membership gives you access to our exclusive 'Buy 1 Get 1 Free' offer on all eyeglasses and sunglasses. It applies to the entire family and is valid for a year."
    },
    {
        "question": "What is your return policy?",
        "answer": "We have a '14-Day No Questions Asked' return policy. If you don't like the fit or style, you can return or exchange them easily."
    },
    {
        "question": "How much do glasses cost?",
        "answer": "Our eyeglasses start from as low as ₹1199 including lenses. The final price depends on the frame brand and the lens package you choose (e.g., Anti-glare, Blu-cut, Progressive)."
    }
]

def load_knowledge_base():
    """Generates FAQ file if missing, then loads it."""
    try:
        path = os.path.join(os.path.dirname(__file__), FAQ_FILE)
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_FAQ, f, indent=4)
        with open(path, "r", encoding='utf-8') as f:
            return json.dumps(json.load(f)) # Return as string for the Prompt
    except Exception as e:
        print(f"⚠️ Error loading FAQ: {e}")
        return ""

STORE_FAQ_TEXT = load_knowledge_base()

# ======================================================
# 💾 2. LEAD DATA STRUCTURE (Eyewear Specific)
# ======================================================

@dataclass
class LeadProfile:
    name: str | None = None
    contact_info: str | None = None # Email or Phone
    product_interest: str | None = None # e.g., Glasses, Sunglasses, Home Checkup
    prescription_status: str | None = None # e.g., Have it, Need checkup, 0 power
    location: str | None = None # City/Area
    timeline: str | None = None # When they want to buy
    
    def is_qualified(self):
        """Returns True if we have minimum contact info"""
        return all([self.name, self.contact_info, self.product_interest])

@dataclass
class Userdata:
    lead_profile: LeadProfile

# ======================================================
# 🛠️ 3. SDR TOOLS
# ======================================================

@function_tool
async def update_lead_profile(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Customer's name")] = None,
    contact_info: Annotated[Optional[str], Field(description="Customer's phone number or email")] = None,
    product_interest: Annotated[Optional[str], Field(description="What they want to buy (Glasses, Sunglasses, Contacts, Eye Test)")] = None,
    prescription_status: Annotated[Optional[str], Field(description="Do they have a prescription or need a checkup?")] = None,
    location: Annotated[Optional[str], Field(description="Customer's city or area (important for home checkup)")] = None,
    timeline: Annotated[Optional[str], Field(description="When they plan to purchase")] = None,
) -> str:
    """
    ✍️ Captures lead details provided by the user during conversation.
    Only call this when the user explicitly provides information.
    """
    profile = ctx.userdata.lead_profile
    
    # Update only fields that are provided (not None)
    if name: profile.name = name
    if contact_info: profile.contact_info = contact_info
    if product_interest: profile.product_interest = product_interest
    if prescription_status: profile.prescription_status = prescription_status
    if location: profile.location = location
    if timeline: profile.timeline = timeline
    
    print(f"📝 UPDATING LEAD: {profile}")
    return "Lead profile updated. Continue the conversation."

@function_tool
async def submit_lead_and_end(
    ctx: RunContext[Userdata],
    email_subject: Annotated[str, Field(description="Subject line for the follow-up email based on conversation context")],
    email_body: Annotated[str, Field(description="The body of the follow-up email (2-3 paragraphs with CTA)")]
) -> str:
    """
    💾 Saves the lead AND the email draft to the database, then signals end of call.
    Call this when the user says goodbye. 
    You MUST generate the email_subject and email_body based on the call context before calling this.
    """
    profile = ctx.userdata.lead_profile
    
    # 1. Save Lead Profile
    lead_db_path = os.path.join(os.path.dirname(__file__), LEADS_FILE)
    lead_entry = asdict(profile)
    lead_entry["timestamp"] = datetime.now().isoformat()
    
    existing_leads = []
    if os.path.exists(lead_db_path):
        try:
            with open(lead_db_path, "r") as f:
                existing_leads = json.load(f)
        except: pass
    
    existing_leads.append(lead_entry)
    with open(lead_db_path, "w") as f:
        json.dump(existing_leads, f, indent=4)

    # 2. Save Email Draft
    email_db_path = os.path.join(os.path.dirname(__file__), EMAILS_FILE)
    email_entry = {
        "lead_name": profile.name,
        "lead_contact": profile.contact_info,
        "subject": email_subject,
        "body": email_body,
        "timestamp": datetime.now().isoformat()
    }
    
    existing_emails = []
    if os.path.exists(email_db_path):
        try:
            with open(email_db_path, "r") as f:
                existing_emails = json.load(f)
        except: pass
        
    existing_emails.append(email_entry)
    with open(email_db_path, "w") as f:
        json.dump(existing_emails, f, indent=4)
        
    print(f"✅ LEAD SAVED TO {LEADS_FILE}")
    print(f"✅ EMAIL DRAFT SAVED TO {EMAILS_FILE}")
    
    # Return instructions to the agent to read out the summary
    return (f"Lead and Email Draft Saved.\n\n"
            f"Subject: {email_subject}\n"
            f"Body Summary: {email_body[:50]}...\n\n"
            f"Tell the user: 'Thanks {profile.name}. I've drafted a follow-up email with details about {profile.product_interest} for you. We'll speak soon!'")

# ======================================================
# 🧠 4. AGENT DEFINITION
# ======================================================

class SDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
            You are 'Riya', a friendly and energetic Sales Development Rep (SDR) for **Lenskart**, India's leading eyewear brand.
            
            📘 **YOUR KNOWLEDGE BASE (FAQ):**
            {STORE_FAQ_TEXT}
            
            🎯 **YOUR GOAL:**
            1. Answer questions about Lenskart's eyewear and services.
            2. **QUALIFY THE LEAD:** Ask for Name, Product Interest, Location, and Contact Info.
            3. **DRAFT FOLLOW-UP:** When the call ends, generate a personalized email draft based on what we discussed.
            
            ⚙️ **BEHAVIOR:**
            - **Be Helpful & Local:** Use a warm, Indian-English professional tone.
            - **Capture Data:** Use `update_lead_profile` immediately when you hear new info.
            
            🔚 **CLOSING PROCEDURE (CRITICAL):**
            When the user says "Goodbye", "That's all", or indicates they are done:
            1. **Mentally draft** a follow-up email.
               - **Subject:** Engaging and relevant (e.g., "Your Lenskart Home Checkup Details").
               - **Body:** 2-3 paragraphs summarizing their interest (e.g., specific frames, eye test) and a Call-To-Action (e.g., "Reply to schedule").
            2. Call `submit_lead_and_end` and pass this `email_subject` and `email_body` into it.
            
            🚫 **RESTRICTIONS:**
            - Do NOT make up fake delivery dates.
            - Ensure the email body is professional and polite.
            """,
            tools=[update_lead_profile, submit_lead_and_end],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "👓" * 25)
    print("🚀 STARTING LENSKART SDR SESSION")
    
    # 1. Initialize State
    userdata = Userdata(lead_profile=LeadProfile())

    # 2. Setup Agent
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-natalie", # Warm professional voice
            style="Promo",        
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )
    
    # 3. Start
    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))