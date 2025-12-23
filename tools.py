from strands import tool, ToolContext


from browse_manager import browser_manager

import logging
logging.basicConfig(level=logging.INFO)

# A tool which helps to open the page and return the page output
@tool(context=True)
async def browse(url: str, tool_context: ToolContext) -> str:
    """
    Open a web page and return its HTML content
    """
    try:
        session_id = tool_context.invocation_state.get("session_id", "asd")
        logging.info(f'Browse session_id {session_id} : {url}')
        page = await browser_manager.get_page(session_id)
        await page.goto(url, wait_until="networkidle")
        return await page.content()
    except Exception as ex:
        logging.error(f'Browse error {ex}')
    return ''