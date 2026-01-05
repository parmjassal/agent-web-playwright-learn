
import logging
from strands import Agent
from strands.tools import tool
from strands.types.content import ContentBlock, Message
from visual_agent import SYSTEM_PROMPT
from visual_agent import llama_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





@tool
def query_image(image_path: str, query: str) :
    """
    Queries visual content from a previously captured screenshot image.

    ─────────────────────────────
    TOOL PURPOSE
    ─────────────────────────────

    - This tool performs READ-ONLY visual analysis on an image file.
    - It is used to extract or verify information that is NOT reliably available from the DOM.
    - Typical use cases:
        - Read visible text from the UI
        - Confirm presence or absence of labels, buttons, warnings, or errors
        - List visible items (e.g., files, table rows, menu options)
        - Validate visual state before choosing selectors or actions

    ─────────────────────────────
    USAGE CONTRACT (STRICT)
    ─────────────────────────────

    IMAGE INPUT:
    - `image_path` MUST be a valid file path on disk
    - The image MUST already exist (usually created by a prior screenshot step)
    - Image format is expected to be PNG

    QUERY INPUT:
    - `query` MUST be a clear, specific, natural-language question
    - The query should focus ONLY on information visible in the image
    - Examples:
        - "List all files visible in the screenshot"
        - "Is there an error message shown on the page?"
        - "What text is displayed on the primary button?"
        - "Is a modal dialog open?"

    EXECUTION RULES:
    - This tool MUST be invoked ONLY after a screenshot step
    - This tool MUST NOT be combined with Playwright actions
    - The tool does NOT modify browser or page state
    - Results are observational evidence only

    RETURN VALUE:
    - On success:
        ToolResponse(status="success", result=<visual analysis output>)
    - On failure:
        ToolResponse(status="error", error=ToolError(...))
    - The final output is a JSON-serialized ToolResponse string

    ERROR HANDLING:
    - Any exception results in:
        - error.code = "IMAGE_QUERY_FAILED"
        - error.type = "RECOVERABLE"

    ─────────────────────────────
    AGENT RESPONSIBILITIES
    ─────────────────────────────

    - The agent MUST:
        - Use the output of this tool as evidence for planning next steps
        - Cross-check image findings with DOM when possible
    - The agent MUST NOT:
        - Assume visual findings change page state
        - Take actions in the same step as image querying

    Args:
        image_path (str):
            Absolute or relative path to the screenshot image file.
        query (str):
            Natural-language question about the image contents.

    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    text_block: ContentBlock = {
        "text": query
    }

    image_block: ContentBlock = {
        "image": {
            "format": "png",
            "source": {
                "bytes": image_bytes
            }
        }
    }

    message: Message = {
        "role": "user",
        "content": [text_block, image_block]
    }
    visual_agent = Agent(
        name="Visual Assistant",
        system_prompt=SYSTEM_PROMPT,
        model=llama_model,
    )
    return visual_agent(prompt=[message])
