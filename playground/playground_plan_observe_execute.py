import json
import logging
from typing import Union, Optional, Dict, Any

from pydantic import BaseModel, Field
from strands.models.llamacpp import LlamaCppModel
from strands_tools.browser import LocalChromiumBrowser
from strands_tools.browser.models import ListLocalSessionsAction, GetHtmlAction, ScreenshotAction, BrowserInput, \
    NavigateAction, InitSessionAction
from strands_tools.python_repl import python_repl

from playgorund.agents import PLANNER_PROMPT, OBSERVER_PROMPT, EXECUTION_PROMPT, SELECTOR_PROMPT
from rate_limit_hook import RateLimitHook
from tool_output_reduction import OutputLimitHook

from strands import Agent, ToolContext
from strands.tools import tool

import gradio as gr

from tools import query_image

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("strands_tools.browser").setLevel(logging.DEBUG)

llama_model = LlamaCppModel(
    base_url="http://localhost:8081",
    # **model_config
    model_id="default",
    params={
        "max_tokens": 10000,
        "temperature": 0.5,
        "repeat_penalty": 1.1,
    }
)


class TestBrowserInput(BaseModel):
    """Input model for browser actions."""

    action: Union[
        ListLocalSessionsAction,
        GetHtmlAction,
        ScreenshotAction,
    ] = Field(discriminator="type")
    wait_time: Optional[int] = Field(default=2, description="Time to wait after action in seconds")

class TestBrowser(LocalChromiumBrowser):

    @tool
    def observe_browser(self, browser_input: TestBrowserInput) -> Dict[str, Any]:
        """
        Browser observation tool for screenshot, html or DOM text or list sessions.

        Args:
            browser_input: Structured input containing the action to perform.

        Returns:
            Dict containing execution results."""

        return self.browser(browser_input)

    async def _async_get_html(self, action: GetHtmlAction) -> Dict[str, Any]:
        """Async get HTML implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            if not action.selector:
                result = await page.content()
            else:
                await page.wait_for_selector(action.selector, timeout=5000)
                result = await page.inner_html(action.selector)
            return {"status": "success", "content": [{"text": result}]}
        except Exception as e:
            logging.debug("exception=<%s> | get HTML action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}


browser = TestBrowser()
browser._default_launch_options = {"persistent_context": True}


import re


@tool(context=True)
def grep_in_html_page(pattern, tool_context: ToolContext = None):
    """
    Find occurrences of a pattern in a string and return matching results
    along with surrounding context.

    This can be used to find elements or text inside an HTML page.
    """
    # make multi-line into single line
    session_id = tool_context.invocation_state["session_id"]
    #session_id = "asd1234567aa"

    text = browser.observe_browser(browser_input=TestBrowserInput(action=GetHtmlAction(type="get_html", session_name=session_id)))
    single_line = re.sub(r'\s+', ' ', text["content"][0]["text"])
    logging.debug("single_line=%s", single_line)
    results = []
    after = 100
    before = 1000
    for match in re.finditer(pattern, single_line):
        start, end = match.span()

        context_start = max(0, start - before)
        context_end = min(len(single_line), end + after)

        results.append({
            "match": match.group(),
            "before": single_line[context_start:start],
            "after": single_line[end:context_end],
            "full": single_line[context_start:context_end],
        })
    logging.debug("results=%s", results)
    return results


SYSTEM_PROMPT = """
System Prompt: Tool-Orchestrating Supervisor
You are a Supervisor agent responsible for achieving the user’s goal by orchestrating external tools.
You do not solve the task directly.You coordinate tools in a structured loop.

Available Tools
* planner: Produces or refines a step-by-step plan.
* observer: Validates whether a plan is feasible and complete, and suggests corrections or missing steps.
* executor: Executes a single concrete action from the plan and reports the result.

Mandatory Control Loop
You must repeatedly perform the following cycle:

Step 1: Planning
Call the planner tool with:
* The current goal
* Any execution results so far
* Any observer feedback
The planner returns a proposed or refined plan.

Step 2: Validation
Call the observer tool with:
* The current plan
* Available tools and constraints
The observer must determine:
* Whether the next step is achievable
* Whether intermediate steps are missing
* Whether the plan needs refinement

Step 3: Plan Refinement
If the observer reports issues:
* Call the planner again using the observer’s feedback
* Do not proceed to execution until the observer approves

Step 4: Execution
Once the observer approves the plan:
* Call the executor tool to execute only the next step
* Capture the execution result verbatim

Loop Rules
* After execution, return to Step 1
* Never execute multiple steps at once
* Never skip validation
* Never fabricate execution results
* Stop only when:
    * The goal is fully achieved, or
    * The observer declares the plan unachievable

Completion Conditions
When the task is complete:
* Provide a concise completion summary
* Do not call further tools
If the task cannot proceed:
* Explain the blocking reason clearly
* Terminate the loop

"""

@tool(context=True)
def planner(goal: str, current_state_summary: str, observer_feedback: str = "", execution_result: str = "", tool_context: ToolContext = None):
    """You are the Planner tool in a multi-agent system.

Your responsibility is to produce a clear, minimal, step-by-step plan
that advances the given goal. The lis should contain all the steps
required to achieve the goal.

Rules:
- Do NOT execute actions.
- Do NOT validate feasibility.
- Assume steps will be validated by a separate Observer.
- Each step must be concrete and actionable.
- Prefer fewer steps unless intermediate steps are logically required.
- Plans may be revised based on feedback.
- Don't take reponsibilities of another tool or agent

Inputs you will receive:
- Goal
- Current state summary
- Observer feedback (if any)
- Execution results (if any)

Output format (mandatory):
PLAN:
1. <step>
2. <step>

Each step must be independently executable.
Do not include explanations outside the plan.
"""
    logging.info(f"Planner: {goal} current_state_summary: {current_state_summary} observer_feedback: {observer_feedback}")
    session_id = tool_context.invocation_state["session_id"]
    agent = Agent(
        name="Planner Agent",
        system_prompt=PLANNER_PROMPT,
        model=llama_model,
        hooks=[RateLimitHook(), OutputLimitHook()]
    )
    prompt = f"goal: {goal} current_state_summary: {current_state_summary} observer_feedback: {observer_feedback} execution_result:{execution_result} session_name:{session_id}"
    ans = agent(prompt, session_id=session_id)
    return ans


@tool(context=True)
def observer(current_step_to_validate: str, executed_steps: str, tool_context: ToolContext = None):
    """You are the Observer tool in a multi-agent system.

Your responsibility is to evaluate a proposed plan and determine
whether it is feasible, complete, and correctly ordered.

Rules:
- Do NOT execute actions.
- Do NOT propose an entirely new plan.
- You may suggest missing prerequisites or intermediate steps.
- You must be conservative: if uncertain, reject and explain.

Validation criteria:
- Each step is achievable with available tools.
- All dependencies are satisfied.
- No logical gaps or hidden assumptions.
- No step bundles multiple actions.

Output format (mandatory):

If the plan is valid:
OBSERVATION:
APPROVED

If the plan is invalid:
OBSERVATION:
REJECTED
Issues:
- <issue>
Suggested additions or corrections:
- <suggested step or constraint>
"""
    logging.info(f"Observer: {current_step_to_validate} executed_steps: {executed_steps}")
    session_id = tool_context.invocation_state["session_id"]
    agent = Agent(
        name="Planner Agent",
        system_prompt=OBSERVER_PROMPT,
        model=llama_model,
        tools=[browser.observe_browser, query_image],
        hooks=[RateLimitHook(), OutputLimitHook()]
    )
    prompt = f"current_step: {current_step_to_validate} executed_steps: {executed_steps} session-name:{session_id}"
    ans = agent(prompt, session_id=session_id)
    return ans


@tool(context=True)
def executor(step_id, step_description, current_state, tool_context: ToolContext = None):
    """Input:
- step_id
- step_description
- current_state

Behavior:
- Execute exactly one step
- Return raw outcome
- Do not interpret success beyond reporting facts

Output:
EXECUTION_RESULT:
- step_id
- action_taken
- result
- status: success | failure
"""
    logging.info(f"Executor: {step_id} step_description: {step_description}")
    session_id = tool_context.invocation_state["session_id"]
    agent = Agent(
        name="Execution Agent",
        system_prompt=EXECUTION_PROMPT,
        model=llama_model,
        tools=[browser.browser, selector],
        hooks=[RateLimitHook(), OutputLimitHook()]
    )
    prompt = f"step_id: {step_id} step_description: {step_description} session-name:{session_id}"
    ans = agent(prompt, session_id=session_id)
    return ans

actions_string = (
    "InitSessionAction, ListLocalSessionsAction, NavigateAction, ClickAction, "
    "TypeAction, EvaluateAction, PressKeyAction, GetTextAction, GetHtmlAction, "
    "ScreenshotAction, RefreshAction, BackAction, ForwardAction, NewTabAction, "
    "SwitchTabAction, CloseTabAction, ListTabsAction, GetCookiesAction, "
    "SetCookiesAction, NetworkInterceptAction, ExecuteCdpAction, CloseAction"
)



@tool(context=True)
def selector(step_description, tool_context: ToolContext = None):
    """Input:
- step_description

Behavior:
- Find the selector which will help to execute the step with high certainty.

Output:
EXECUTION_RESULT:
- selector
- confidence_score
"""
    logging.info(f"Selector: Finding selector for step_description: {step_description}")
    session_id = tool_context.invocation_state["session_id"]
    agent = Agent(
        name="Selector Agent",
        system_prompt=SELECTOR_PROMPT,
        model=llama_model,
        tools=[query_image, grep_in_html_page],
        hooks=[RateLimitHook(), OutputLimitHook()]
    )
    prompt = f"step_description: {step_description} allowed-actions-for-step:{actions_string} session-name:{session_id}"
    ans = agent(prompt, session_id=session_id)
    return ans




agent = Agent(
    name="Orchestrator Agent",
    system_prompt=SYSTEM_PROMPT,
    model=llama_model,
    tools=[planner, observer, executor],
    hooks=[RateLimitHook(), OutputLimitHook()]
)


async def chat(message, _, request: gr.Request):
    try:
        # Execute the agent
        result = agent(message, session_id=request.session_hash)
        return str(result)
    except Exception as e:
        logging.exception("Agent error")
        return f"Error: {str(e)}"

# Launch Gradio
gr.ChatInterface(chat).launch()

#if __name__ == '__main__':

#    browser_input = BrowserInput(action=InitSessionAction(session_name="asd1234567aa" , description="", type="init_session"))

#    browser.browser(browser_input=BrowserInput(action=InitSessionAction(session_name="asd1234567aa" , description="", type="init_session")))
#    browser.browser(browser_input=BrowserInput(action=NavigateAction(session_name="asd1234567aa", url="https://www.miniindia.ie", type="navigate") ))
#    grep_in_html_page("Eircode")
#    print(agent("Goal: Add Coke to Cart from miniindia.ie Other Data: EIRCODE=K78A4E4 PreferredDelivery=Collection", session_id="miniindia-example"))
