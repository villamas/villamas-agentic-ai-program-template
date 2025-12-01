import operator
import json
import os
from typing import Annotated, List, TypedDict, Union
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Define the State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    plan: List[str]
    current_step: int
    results: dict

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# --- Planner Node ---
planner_prompt = ChatPromptTemplate.from_template(
    """You are a planner. Given a user request, create a step-by-step plan to answer it.
    Return the plan as a JSON list of strings.
    
    Request: {request}
    
    Plan:"""
)

def planner_node(state: AgentState):
    print("--- Planner Node ---")
    request = state["messages"][-1].content
    chain = planner_prompt | llm
    response = chain.invoke({"request": request})
    
    try:
        # Clean up markdown code blocks if present
        content = response.content.replace("```json", "").replace("```", "").strip()
        plan = json.loads(content)
    except json.JSONDecodeError:
        # Fallback
        plan = [line.strip() for line in response.content.split('\n') if line.strip()]
    
    print(f"Generated Plan: {plan}")
    return {"plan": plan, "current_step": 0, "results": {}}

# --- Executor Node ---
executor_prompt = ChatPromptTemplate.from_template(
    """You are an executor. Execute the following step: {step}.
    Previous results: {results}
    
    If the step requires information you don't have, simulate a reasonable search result or calculation.
    
    Return the result of the execution."""
)

def executor_node(state: AgentState):
    print("--- Executor Node ---")
    plan = state["plan"]
    current_step = state["current_step"]
    
    if current_step >= len(plan):
        return {"messages": [AIMessage(content="All steps completed.")]}
        
    step = plan[current_step]
    results = state["results"]
    
    print(f"Executing Step {current_step + 1}: {step}")
    
    chain = executor_prompt | llm
    response = chain.invoke({"step": step, "results": results})
    
    # Update results
    results[f"step_{current_step+1}"] = response.content
    
    return {
        "current_step": current_step + 1,
        "results": results,
        "messages": [AIMessage(content=f"Executed: {step} -> {response.content}")]
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
        END: END
    }
)

app = workflow.compile()

# --- Main Execution ---
if __name__ == "__main__":
    # Example query
    user_input = "Research the population of Tokyo and New York, then calculate the difference."
    
    inputs = {
        "messages": [HumanMessage(content=user_input)]
    }
    
    print(f"User Request: {user_input}")
    
    for event in app.stream(inputs):
        pass # Output is handled by print statements in nodes
        
    print("\n--- Final Results ---")
    # We can't easily access the final state from stream loop directly without capturing it,
    # but the nodes printed progress.
