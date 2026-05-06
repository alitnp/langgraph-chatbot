from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatOllama(model="qwen2.5:14b", temperature=0)

tavily_search_tool = TavilySearch(max_results=3, topic="general")
tools = [tavily_search_tool]
llm_with_tools = llm.bind_tools(tools)


class BasicMassageState(TypedDict):
    messages: Annotated[List, add_messages]


GENERATE_NODE = "generate"
TOOL_NODE = "tools"
TOOL_ROUTER = "tool_router"


def generate_response(state: BasicMassageState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def tool_router(state: BasicMassageState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return TOOL_NODE
    else:
        return END


memory = MemorySaver()
graph = StateGraph(BasicMassageState)
config = {"configurable": {"thread_id": "basic_chatbot_thread"}}


graph.add_node(GENERATE_NODE, generate_response)
graph.add_node(TOOL_NODE, ToolNode(tools))
graph.add_conditional_edges(GENERATE_NODE, tool_router)
graph.add_edge(TOOL_NODE, GENERATE_NODE)
graph.set_entry_point(GENERATE_NODE)

app = graph.compile(checkpointer=memory)


if __name__ == "__main__":
    while True:
        user_input = input("user: ")
        if user_input.lower() in ["exit", "end"]:
            break
        else:
            res = app.stream(
                {"messages": HumanMessage(content=user_input)}, config=config
            )
            for event in res:
                for value in event.values():
                    if value:
                        value["messages"][-1].pretty_print()
