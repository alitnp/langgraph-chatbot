import sqlite3
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()
sql_db = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
llm = ChatOllama(model="qwen2.5:14b", temperature=0)

tavily_search_tool = TavilySearch(max_results=3, topic="general")

tools = [tavily_search_tool]

llm_with_tools = llm.bind_tools(tools)


class BasicMessage(TypedDict):
    messages: Annotated[List, add_messages]


#  node names
GENERATOR = "generator"
TOOLS = "tools"
TOOL_ROUTER = "tool_router"


def generate(state: BasicMessage) -> BasicMessage:
    return {"messages": llm_with_tools.invoke(state["messages"])}


def tool_router(state: BasicMessage):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return TOOLS
    else:
        return END


tool_node = ToolNode(tools)

graph = StateGraph(BasicMessage)
graph.add_node(GENERATOR, generate)
graph.add_node(TOOLS, tool_node)
graph.add_conditional_edges(GENERATOR, tool_router)
graph.add_edge(TOOLS, GENERATOR)
graph.set_entry_point(GENERATOR)

# memory = SqliteSaver(sql_db)
config = {"configurable": {"thread_id": "1"}}

conn = sqlite3.connect(
    "checkpoint.sqlite", check_same_thread=False, isolation_level=None
)

memory = SqliteSaver(conn)
# memory.setup()
app = graph.compile(checkpointer=memory)

while True:
    user_input = input("user: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    else:
        # Stream the graph events
        for event in app.stream({"messages": [("user", user_input)]}, config):
            # Print the event values
            for value in event.values():
                messages_update = value["messages"]

                # The update can be either a list of messages or a single message object
                if isinstance(messages_update, list):
                    messages_update[-1].pretty_print()
                else:
                    messages_update.pretty_print()


sql_db.close()
