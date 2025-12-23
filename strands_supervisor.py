import logging
logging.basicConfig(level=logging.INFO)


from strands import Agent
from strands.models import BedrockModel
from tools import browse

from strands.session.file_session_manager import FileSessionManager

SYSTEM_PROMPT = """
You are an assistant that helps browse the web based on user intent.
Use the available tools when necessary and explain your reasoning.
Always pass the session_id parameter when calling browsing tools.
"""

bedrock_model = BedrockModel(
    model_id="qwen.qwen3-next-80b-a3b",
    region_name="us-east-1",
    temperature=0.5,
)

agents = {}
agent = Agent(name="Browser Controller Agent",system_prompt=SYSTEM_PROMPT,model=bedrock_model,tools=[browse])


import gradio as gr


def consume_stream(stream):
    chunks = []
    for event in stream:
        if isinstance(event, str):
            chunks.append(event)
        elif hasattr(event, "content"):
            chunks.append(event.content)
    return "".join(chunks)


def chat(message, history, request: gr.Request):
    try:
        result = agent(message, session_id=request.session_hash)
        return str(result)
    except Exception as e:
        logging.exception("Agent error")
        return f"Error: {str(e)}"

gr.ChatInterface(chat).launch()