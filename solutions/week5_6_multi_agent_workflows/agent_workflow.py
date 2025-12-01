import operator
import json
from typing import Annotated, List, TypedDict, Dict, Any

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun


# Load environment variables (expects OPENAI_API_KEY in .env)
load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    plan: List[str]
    current_step: int
    results: Dict[str, Any]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize tools
search_tool = DuckDuckGoSearchRun()


# --- Planner Node ---

planner_prompt = ChatPromptTemplate.from_template(
    """You are a planner. Given a user request, create a step-by-step plan to answer it.
Return the plan as a JSON list of strings.

Request: {request}

Plan:"""
)


def planner_node(state: AgentState) -> dict:
    print("--- Planner Node ---")
    request = state["messages"][-1].content
    chain = planner_prompt | llm
    response = chain.invoke({"request": request})

    try:
        # Remove possible markdown code fences
        content = (
            response.content.replace("```json", "")
            .replace("```", "")
            .strip()
        )
        plan = json.loads(content)
    except json.JSONDecodeError:
        plan = [line.strip() for line in response.content.split("\n") if line.strip()]

    print(f"Generated Plan: {plan}")
    return {"plan": plan, "current_step": 0, "results": {}}


# --- Executor Node ---

executor_prompt = ChatPromptTemplate.from_template(
    """You are an executor. Execute the following step: {step}.
Previous results: {results}

If the step requires information you don't have, simulate a reasonable search result or calculation.

Return the result of the execution."""
)


def executor_node(state: AgentState) -> dict:
    print("--- Executor Node ---")
    plan = state["plan"]
    current_step = state["current_step"]

    if current_step >= len(plan):
        return {"messages": [AIMessage(content="All steps completed.")]}

    step = plan[current_step]
    results = dict(state["results"])

    print(f"Executing Step {current_step + 1}: {step}")

    step_lower = step.lower()

    # Use DuckDuckGoSearchRun for research-like steps
    if "population" in step_lower or "research" in step_lower or "search" in step_lower:
        search_query = step
        try:
            search_result = search_tool.invoke(search_query)
        except Exception as e:
            search_result = f"Error while searching: {e!r}"

        result_text = f"[SEARCH RESULT] {search_result}"

    # Use LLM to calculate differences based on previous results
    elif "difference" in step_lower or "calculate" in step_lower or "subtract" in step_lower:
        context = "\n".join(f"{k}: {v}" for k, v in results.items())

        calc_prompt = ChatPromptTemplate.from_template(
            """You are a calculator agent.

Previous step results:
{context}

Instruction: {step}

Use the numerical information above to compute the answer, then explain briefly."""
        )

        chain = calc_prompt | llm
        response = chain.invoke({"context": context, "step": step})
        result_text = response.content

    # Generic fallback using the executor prompt
    else:
        chain = executor_prompt | llm
        response = chain.invoke({"step": step, "results": results})
        result_text = response.content

    results[f"step_{current_step + 1}"] = result_text

    return {
        "current_step": current_step + 1,
        "results": results,
        "messages": [AIMessage(content=f"Executed: {step} -> {result_text}")],
    }


# --- Graph Construction ---

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")


def should_continue(state: AgentState):
    if state["current_step"] < len(state["plan"]):
        return "executor"
    return END


workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "executor": "executor",
        END: END,
    },
)

app = workflow.compile()


# --- Main Execution ---

if __name__ == "__main__":
    user_input = "Research the population of Tokyo and New York, then calculate the difference."

    inputs = {
        "messages": [HumanMessage(content=user_input)],
    }

    print(f"User Request: {user_input}")

    # Stream events to see node-by-node progress
    for event in app.stream(inputs):
        for node_name, value in event.items():
            print(f"\n[Event from node: {node_name}]")
            if "plan" in value:
                print("  Plan:", value["plan"])
            if "messages" in value and value["messages"]:
                print("  Message:", value["messages"][-1].content)

    # Get final state to see all step results
    final_state = app.invoke(inputs)
    print("\n--- Final Results ---")
    print(final_state.get("results", {}))
