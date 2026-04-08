# Standard library imports
import os

# Type annotation utilities
from typing import Annotated
from typing_extensions import TypedDict

# LangGraph message utility that merges new messages into existing state
# instead of replacing them — essential for maintaining chat history
from langgraph.graph.message import add_messages

# Ollama-backed LLM integration (runs models locally via Ollama)
from langchain_ollama import ChatOllama

# Core graph building blocks: the graph class itself plus sentinel start/end nodes
from langgraph.graph import StateGraph, START, END

# Prebuilt node that automatically dispatches tool calls made by the LLM,
# and a routing condition that decides whether to call a tool or end the turn
from langgraph.prebuilt import ToolNode, tools_condition

# Decorator that turns a plain Python function into a LangChain tool
from langchain_core.tools import tool

# A special message type injected at the top of every prompt to set LLM behaviour
from langchain_core.messages import SystemMessage


# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------

# Initialise the local Qwen 2.5 (7B) model via Ollama.
# temperature=0 makes responses deterministic / less creative — good for
# code generation and command execution where accuracy matters more than variety.
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
)


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class State(TypedDict):
    """
    Defines the shape of the data that flows through every node in the graph.

    `messages` accumulates the full conversation history.
    The `Annotated[list, add_messages]` hint tells LangGraph to *append*
    new messages to the list rather than overwrite it on each state update.
    """
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def run_command(cmd: str):
    """
    Takes a command line prompt and executes it on the user's machine and 
    returns the output of the command.
    Example: run_command(cmd="ls") where ls is the command to list the files.
    """
    # Execute the shell command directly on the host machine.
    # os.system returns the exit code (0 = success, non-zero = error).
    result = os.system(command=cmd)
    return result


# ---------------------------------------------------------------------------
# LLM + Tool Binding
# ---------------------------------------------------------------------------

# Attach the tool list to the LLM so it knows it *can* call run_command.
# When the model decides a tool is needed it emits a structured tool-call
# object, which the ToolNode below will actually execute.
llm_with_tool = llm.bind_tools(tools=[run_command])


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def chatbot(state: State):
    """
    Primary reasoning node — called every time the graph needs the LLM to think.

    It prepends a system prompt to the conversation history and invokes the
    tool-aware LLM. The model either:
      • returns a plain text reply  →  the conversation ends this turn, OR
      • emits a tool-call object    →  the graph routes to the 'tools' node next.
    """
    # System prompt defines the assistant's persona and constraints.
    # Keeping it inside the node means it's always prepended fresh, which avoids
    # accidentally duplicating it across turns.
    system_prompt = SystemMessage(content="""
        You are an AI Coding assistant who takes an input from user and based on available
        tools you choose the correct tool and execute the commands.
                                  
        You can even execute commands and help user with the output of the command.

        Always make sure to keep your generated codes and files in chat_gpt/ folder. you can create one if not already there.                           
    """)

    # Combine the system prompt with the full message history and call the LLM.
    message = llm_with_tool.invoke([system_prompt] + state["messages"])

    # Wrap the reply in a list so LangGraph's add_messages reducer can append it
    # to the existing message list in state.
    return {"messages": [message]}


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

# ToolNode wraps run_command so LangGraph can call it automatically whenever
# the LLM emits a matching tool-call in its response.
tool_node = ToolNode(tools=[run_command])

# Create the graph, telling it what the state schema looks like.
graph_builder = StateGraph(State)

# Register both processing nodes with the graph.
graph_builder.add_node("chatbot", chatbot)   # LLM reasoning step
graph_builder.add_node("tools", tool_node)   # Tool execution step

# Every conversation starts at the chatbot node.
graph_builder.add_edge(START, "chatbot")

# After the chatbot node runs, use the built-in `tools_condition` router:
#   • If the LLM's last message contains a tool call  →  route to "tools"
#   • Otherwise                                        →  route to END
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# After a tool finishes executing, always loop back to the chatbot so the LLM
# can reason about the tool's output and decide what to do next.
graph_builder.add_edge("tools", "chatbot")

# Explicit terminal edge: once the chatbot decides no more tools are needed,
# the turn is complete.
graph_builder.add_edge("chatbot", END)


# ---------------------------------------------------------------------------
# Graph Factory
# ---------------------------------------------------------------------------

def create_chat_graph(checkpointer):
    """
    Compile and return the runnable graph with persistence enabled.

    `checkpointer` is injected here (rather than hard-coded) so the caller
    can swap in different storage backends (e.g. in-memory for tests,
    SQLite/Postgres for production) without touching this file.
    """
    return graph_builder.compile(checkpointer=checkpointer)
