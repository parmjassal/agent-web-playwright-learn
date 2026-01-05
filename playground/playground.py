import logging

from strands.models.llamacpp import LlamaCppModel

from rate_limit_hook import RateLimitHook
from tool_output_reduction import OutputLimitHook

from strands import Agent
from strands.tools import tool

logging.basicConfig(level=logging.INFO)

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

@tool
def planner(goal: str, current_state_summary: str, observer_feedback: str = None, execution_Result: str = None):
    """You are the Planner tool in a multi-agent system.

Your responsibility is to produce a clear, minimal, step-by-step plan
that advances the given goal.

Rules:
- Do NOT execute actions.
- Do NOT validate feasibility.
- Assume steps will be validated by a separate Observer.
- Each step must be concrete and actionable.
- Prefer fewer steps unless intermediate steps are logically required.
- Plans may be revised based on feedback.

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
    if goal.startswith("Order") and (observer_feedback is None or observer_feedback == ""):
        logging.info("Returning first step")
        return {"PLAN": {"step-1": "Add Coke to Cart"}}

    if observer_feedback.find("open") !=-1:
        return {"PLAN": {"step-1": "Go to miniindia.ie" , "step-2": "Add Coke to Cart"}}

    return {"PLAN": {"step-2": "Fill the EIRCode", "step-3": "Selection collection as delivery type", "step-4":"Submit the form", "step-6":"Add Coke to Cart"}}



@tool
def observer(current_step_to_validate: str, executed_steps: str):
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
    if current_step_to_validate.startswith("Add") and (executed_steps is None or len(executed_steps) == 0):
        return {"OBSERVATION": "REJECTED", "Issues": [
        "Browser tab is empty"], "corrections": ["Open the required page"]}
    if current_step_to_validate.startswith("Add") and executed_steps.find("miniindia") !=-1 and executed_steps.find("EIRCode") ==-1:
        return {"OBSERVATION": "REJECTED", "Issues": [
        "Modal is hiding page"], "corrections": ["Fill EIRCode, select delivery as collection and submit"]}
    else:
        return {"OBSERVATION": "APROVED"}


@tool
def executor(step_id, step_description, current_state):
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
- artifacts (optional)
"""
    logging.info(f"Executor: {step_id} step_description: {step_description}")
    return {
        "EXECUTION_RESULT": {"step_id": step_id, "action_taken": step_description, "result": step_description + " Done",
                             "status": "success"}}


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

agent = Agent(
    name="Orchestrator Agent",
    system_prompt=SYSTEM_PROMPT,
    model=llama_model,
    tools=[planner, observer, executor],
    hooks=[RateLimitHook(), OutputLimitHook()]
)




if __name__ == '__main__':

    print(agent("current_state_summary: Current page is empty. Goal: Order coke from miniindia.ie"))
