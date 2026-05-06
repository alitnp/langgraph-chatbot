import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from common.llms import ollama_llm

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


tavily_search_tool = TavilySearch(max_results=3, return_sources=True)
tools = [tavily_search_tool]
llm = ollama_llm
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: ChatState) -> ChatState:
    print("ChatState:", len(state["messages"]))
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def tool_router(state: ChatState) -> ChatState:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END


tool_node = ToolNode(tools=tools)


graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)
graph.add_conditional_edges("chatbot", tool_router)
graph.add_edge("tool_node", "chatbot")
graph.set_entry_point("chatbot")

app = graph.compile(checkpointer=MemorySaver())

# (Optional) Visualize the graph
display(Image(app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)))

# 2. Automatically generate a unique session ID for this script run
session_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": session_id}}

print("Chatbot started! Type 'exit' to quit.")


while True:
    user_input = input("user: ")
    if user_input.lower() in {"exit", "quit"}:
        break
    else:
        # Stream the graph events
        for event in app.stream({"messages": [("user", user_input)]}, config):
            # Print the event values
            for value in event.values():
                value["messages"][-1].pretty_print()
