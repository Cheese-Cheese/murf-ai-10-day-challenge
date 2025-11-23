import json
import logging
import os
import sys

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from typing import Annotated, Literal, List
from pydantic import BaseModel, Field

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# --- Coffee Order State ---
class CoffeeOrder(BaseModel):
    """The current state of the coffee order."""
    drinkType: Annotated[
        str,
        Field(
            description="The type of coffee drink, e.g., 'Latte', 'Cappuccino', 'Americano'. MUST be one of the suggested types or a generic 'Coffee'.",
            json_schema_extra={"enum": ["Latte", "Cappuccino", "Americano", "Espresso", "Mocha", "Drip Coffee", "Tea"]}
        )
    ]
    size: Annotated[
        Literal["Small", "Medium", "Large"],
        Field(description="The requested size of the drink.")
    ]
    milk: Annotated[
        Literal["Whole", "Skim", "Oat", "Almond"],
        Field(description="The type of milk for the drink.")
    ]
    extras: Annotated[
        List[str],
        Field(description="Any extra additions, e.g., 'Extra shot', 'Vanilla syrup', 'Whipped cream'.")
    ] = []
    name: Annotated[
        str,
        Field(description="The customer's first name, for the order.")
    ]

# --- Agent Definition ---
class BaristaAgent(Agent):
    def __init__(self) -> None:
        instructions = """You are **TinMaster**, a friendly and quick barista at **Coff Cafe**. 
            The user is speaking to you via voice and wants to place an order.
            You should first introuduce yourslf when the user greets you. 
            Your responses should be upbeat, concise, and focused on gathering the complete order details.
            
            **Your Goal:** Complete the customer's order by filling all fields of the CoffeeOrder object.
            
            1.  **Ask for Missing Details:** If any field in the CoffeeOrder is missing (drinkType, size, milk, or name), you **must** ask a clarifying question for that specific field. For example: "What size would you like that?" or "And what's your name for the order?"
            2.  **Confirm Extras:** If the user mentions any extras (like 'extra shot', 'vanilla syrup', 'whip'), add them to the 'extras' list.
            3.  **Finalize Order:** Once all fields are filled, you **MUST** call the `save_order_to_file` tool to complete the transaction and give a friendly closing line.
            4.  **Stay in Character:** Use light, friendly language appropriate for a coffee shop. Avoid technical jargon or complex formatting.
        """
        super().__init__(
            instructions=instructions,
            tools=[]
        )
        self.order_id_counter = 0

    @function_tool
    async def save_order_to_file(self, ctx: RunContext, order: CoffeeOrder) -> str:
        """
        Call this tool **ONLY** when the entire CoffeeOrder object is fully specified (drinkType, size, milk, and name are all filled). 
        This finalizes the order and generates the receipt.
        """
        self.order_id_counter += 1
        
        # Create a simple JSON file for the order summary
        order_summary = {
            "order_number": f"CC-{self.order_id_counter:03d}",
            "customer_name": order.name,
            "drink": f"{order.size} {order.drinkType} with {order.milk} milk",
            "extras": order.extras if order.extras else "None",
            "status": "Complete"
        }
        
        # In a real application, you'd save this to a database. Here we just log and return.
        file_name = f"order_{order_summary['order_number']}.json"
        with open(file_name, 'w') as f:
            json.dump(order_summary, f, indent=2)

        return f"Order {order_summary['order_number']} for {order.name} has been placed. Details saved to {file_name}. Order JSON: {json.dumps(order_summary)}"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with the BaristaAgent
    await session.start(
        agent=BaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))