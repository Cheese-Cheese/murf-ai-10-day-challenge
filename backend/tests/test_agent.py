import pytest
from livekit.agents import AgentSession, inference, llm
from livekit.agents.chat import ChatContext
from agent import BaristaAgent  # Import the new agent class


def _llm() -> llm.LLM:
    # Use a judge model that is good at intent assessment
    return inference.LLM(model="openai/gpt-4o") 


@pytest.mark.asyncio
async def test_barista_greeting() -> None:
    """Evaluation of the Barista agent's friendly greeting and persona."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(BaristaAgent())

        # Run an agent turn following the user's greeting
        result = await session.run(user_input="Hi, I'd like a coffee.")

        # Evaluate the agent's response for a friendly, in-character greeting and initial question
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                Greets the user in a friendly Barista manner and asks for the specific drink 
                (e.g., Latte, Cappuccino) or the customer's name, as the first step in taking an order.
                """,
            )
        )

        # Ensures there are no function calls or other unexpected events at the start
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_order_completion_and_tool_call() -> None:
    """Evaluation of the agent's ability to gather all order details and call the save tool."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as sess,
    ):
        agent = BaristaAgent()
        await sess.start(agent)

        # 1. Start order
        result = await sess.run(user_input="Can I get a large Latte?")
        # Agent should ask for milk type and name (or one of them)
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Should ask for the missing milk type and/or customer name."
            )
        )
        result.expect.no_more_events()
        
        # 2. Add milk and an extra
        result = await sess.run(user_input="Oat milk please, with an extra shot.")
        # Agent should now ask for the missing customer name
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Should ask for the customer's name."
            )
        )
        result.expect.no_more_events()

        # 3. Finalize order with the name
        result = await sess.run(user_input="It's for Alex.")
        result.expect.skip_next_event_if(type="message", role="assistant") # Skip initial confirmation
        
        # Check for the function call to save the order
        func_call = result.expect.next_event().is_function_call(
            name="save_order_to_file"
        )
        
        # Validate arguments of the function call
        expected_arguments = {
            "drinkType": "Latte",
            "size": "Large",
            "milk": "Oat",
            "extras": ["Extra shot"],
            "name": "Alex",
        }
        
        # Use a judge for a flexible check of the arguments
        await func_call.judge(
            llm, 
            intent=f"The arguments must match the completed order: {expected_arguments}"
        )

        # Check for the function call output
        result.expect.next_event().is_function_call_output()
        
        # Check for the final confirmation message
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Should confirm the order is placed and thank the customer."
            )
        )
        result.expect.no_more_events()