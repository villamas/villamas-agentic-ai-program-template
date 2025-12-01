# Lab: Multi-Agent Workflows with LangGraph

In this lab, you will build a multi-agent system using **LangGraph**. You will implement a **Planner-Executor** architecture where one agent creates a plan and another executes it step-by-step.

## Prerequisites

- Python 3.10+
- OpenAI API Key (or compatible LLM key)

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file in this directory:
   ```env
   OPENAI_API_KEY=sk-...
   ```

## Step 1: Define the State

In LangGraph, the **State** is the shared memory of the graph. We need to define what information our agents will pass around.

Create a file named `agent_workflow.py` and add:

```python
import operator
from typing import Annotated, List, TypedDict, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    plan: List[str]
    current_step: int
    results: dict
```

- `messages`: Chat history.
- `plan`: A list of steps to execute.
- `current_step`: Which step we are on.
- `results`: Storing outputs from steps.

## Step 2: Create the Planner Agent

The Planner's job is to take a user request and break it down into a list of steps.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

llm = ChatOpenAI(model="gpt-4o-mini")

planner_prompt = ChatPromptTemplate.from_template(
    """You are a planner. Given a user request, create a step-by-step plan to answer it.
    Return the plan as a JSON list of strings.
    
    Request: {request}
    
    Plan:"""
)

def planner_node(state: AgentState):
    request = state["messages"][-1].content
    chain = planner_prompt | llm
    response = chain.invoke({"request": request})
    
    # Parse the JSON plan (simplified for demo)
    # In production, use structured output or Pydantic parsers
    try:
        plan = json.loads(response.content)
    except:
        # Fallback if LLM doesn't return pure JSON
        plan = [line.strip() for line in response.content.split('\n') if line.strip()]
    
    return {"plan": plan, "current_step": 0, "results": {}}
```

## Step 3: Create the Executor Agent

The Executor takes the current step and performs it. For this lab, we will simulate execution or use simple tools.

```python
executor_prompt = ChatPromptTemplate.from_template(
    """You are an executor. Execute the following step: {step}.
    Previous results: {results}
    
    Return the result of the execution."""
)

def executor_node(state: AgentState):
    plan = state["plan"]
    current_step = state["current_step"]
    
    if current_step >= len(plan):
        return {"messages": [AIMessage(content="All steps completed.")]}
        
    step = plan[current_step]
    results = state["results"]
    
    chain = executor_prompt | llm
    response = chain.invoke({"step": step, "results": results})
    
    # Update results
    results[f"step_{current_step+1}"] = response.content
    
    return {
        "current_step": current_step + 1,
        "results": results,
        "messages": [AIMessage(content=f"Executed: {step} -> {response.content}")]
    }
```

## Step 4: Build the Graph

Now we connect the nodes.

```python
from langgraph.graph import StateGraph, END

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
```

## Step 5: Run the Workflow

```python
if __name__ == "__main__":
    inputs = {
        "messages": [HumanMessage(content="Research the population of Tokyo and New York, then calculate the difference.")]
    }
    
    for event in app.stream(inputs):
        for key, value in event.items():
            print(f"\nNode '{key}':")
            if "plan" in value:
                print(f"  Plan: {value['plan']}")
            if "messages" in value:
                print(f"  Message: {value['messages'][-1].content}")
```

## Challenge

Modify the `executor_node` to use actual tools (like `DuckDuckGoSearchRun` or a custom Python function) instead of just asking the LLM to simulate it.
