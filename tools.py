

from strands import tool, ToolContext, Agent
from strands.types.tools import ToolUse

from browse_manager import browser_manager

import logging

logging.basicConfig(level=logging.INFO)


def browse_sync(url: str, session_id: str) -> str:
    page = browser_manager.new_page_sync(session_id)

    try:
        asyncio.run_coroutine_threadsafe(
            page.goto(url),
            browser_manager.loop
        ).result()

        content = asyncio.run_coroutine_threadsafe(
            page.content(),
            browser_manager.loop
        ).result()

        return content

    finally:
        browser_manager.close_page_sync(page)

# A tool which helps to open the page and return the page output
@tool(context=True)
def browse(url: str, tool_context: ToolContext) -> str:
    """
    Open a web page and return its HTML content
    """

    try:
        logging.info(f"opening {url}")
        session_id = tool_context.invocation_state.get("session_id", "asd")
        return browse_sync(url, session_id)
    except Exception as ex:
        logging.error(f'Browse error {ex}')
    return ''

import asyncio

def main():
    tool_context = ToolContext(tool_use=ToolUse(input="", name="browse", toolUseId="asd"), agent=Agent(),
                               invocation_state={"session_id": "asd"})
    tool_context_1 = ToolContext(tool_use=ToolUse(input="", name="browse", toolUseId="asd"), agent=Agent(),
                               invocation_state={"session_id": "asda"})
    browse("https://parmjassal.github.io/learnings/lightweight-e2e-testing-for-flink/", tool_context)
    browse("https://parmjassal.github.io/learnings/lightweight-e2e-testing-for-flink/", tool_context_1)
    browse("https://parmjassal.github.io/learnings/lightweight-e2e-testing-for-flink/", tool_context)
    browse("https://parmjassal.github.io/learnings/lightweight-e2e-testing-for-flink/", tool_context_1)



if __name__ == "__main__":
    main()