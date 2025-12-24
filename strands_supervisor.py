import logging
import gradio as gr
from strands.models import BedrockModel

from rate_limit_hook import RateLimitHook
from tools import browse

from strands import ToolContext, Agent
from strands.types.tools import ToolUse

logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """
You are an assistant that helps browse the web based on user intent.
Use the available tools when necessary and explain your reasoning.
Always pass the session_id parameter when calling browsing tools.
"""

# Initialize models globally (Safe, as they are usually HTTP clients)
bedrock_model = BedrockModel(
    model_id="qwen.qwen3-next-80b-a3b",
    region_name="us-east-1",
    temperature=0.5,
)

agent = Agent(
    name="Browser Controller Agent",
    system_prompt=SYSTEM_PROMPT,
    model=bedrock_model,
    tools=[browse],
    hooks=[RateLimitHook()]
)


async def chat(message, history, request: gr.Request):
    try:
        # Execute the agent
        result = agent(message, session_id=request.session_hash)
        return str(result)
    except Exception as e:
        logging.exception("Agent error")
        return f"Error: {str(e)}"

# Launch Gradio
gr.ChatInterface(chat).launch()
