import sqlite3
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain.messages import SystemMessage
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
    route_to: Annotated[str, "route_to"]


#  node names
GENERAL_GENERATOR = "general_generator"
POST_GENERATOR = "post_generator"
TOOLS = "tools"
TOOL_ROUTER = "tool_router"
INTENT_CLASSIFIER = "intent_classifier"
SELECT_WORKER = "select_worker"
HUMAN_REVIEW = "human_review"


def post_generator(state: BasicMessage) -> BasicMessage:
    print("Post-processing response...")
    # add a system message to strongly encourage tool usage for current events
    system_message = SystemMessage(
        content=(
            "You are an expert Social Media Post Generator. "
            "You MUST use the provided `tavily_search_results_json` tool to search the internet for ANY factual claims or current events requested by the user. "
            "DO NOT say 'I will search' or 'Let me check'. You MUST output the exact JSON tool call required to invoke the search tool immediately. "
            "Once you receive the tool's output, then write the post."
        )
    )
    messages = [system_message] + state["messages"]
    return {"messages": llm_with_tools.invoke(messages)}


def general_generator(state: BasicMessage) -> BasicMessage:
    print("Generating general response...")
    system_message = SystemMessage(
        content="You are a helpful assistant that provides information and engages in general conversation with users. You can use the tools at your disposal to find information if you don't know the answer or you information is outdated, but you do not need to use them for every response. Focus on providing clear and informative answers to the user's questions, and engaging in a friendly and helpful manner."
    )
    messages = [system_message] + state["messages"]
    return {"messages": llm_with_tools.invoke(messages)}


def intent_classifier(state: BasicMessage) -> BasicMessage:
    print("Classifying user intent...")
    user_message = state["messages"][-1]
    classification_propmpt = f"""Classify the intent of the user's message: 
    {user_message.content}
    if the user is asking you to write a social media post or improve an existing one, return exactly {POST_GENERATOR}.
    and if they are asking for general information or engaging in casual conversation, return exactly {GENERAL_GENERATOR}."""
    response = llm.invoke(classification_propmpt)
    intent = response.content.strip().lower()
    print(f"Classified intent: {intent}")
    if POST_GENERATOR.lower() in intent:
        route_to = POST_GENERATOR
    else:
        route_to = GENERAL_GENERATOR
    return {"route_to": route_to}


def tool_router(state: BasicMessage):
    print("Routing to tools if necessary...")
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return TOOLS
    else:
        # Route to HUMAN_REVIEW instead of END
        return HUMAN_REVIEW


def node_router(state: BasicMessage):
    print("Routing to next node...")
    route_to = state["route_to"]
    if route_to == POST_GENERATOR:
        return POST_GENERATOR
    else:
        return GENERAL_GENERATOR


def human_review_node(state: BasicMessage) -> BasicMessage:
    # return an empty dict to indicate no state changes were made
    return {}


graph = StateGraph(BasicMessage)
graph.add_node(GENERAL_GENERATOR, general_generator)
graph.add_node(POST_GENERATOR, post_generator)
graph.add_node(TOOLS, ToolNode(tools))
graph.add_node(INTENT_CLASSIFIER, intent_classifier)
graph.add_node(HUMAN_REVIEW, human_review_node)
graph.add_edge(HUMAN_REVIEW, END)

graph.add_conditional_edges(GENERAL_GENERATOR, tool_router)
graph.add_conditional_edges(POST_GENERATOR, tool_router)
graph.add_conditional_edges(TOOLS, node_router)
graph.add_conditional_edges(INTENT_CLASSIFIER, node_router)


graph.set_entry_point(INTENT_CLASSIFIER)

memory = SqliteSaver(sql_db)

app = graph.compile(checkpointer=memory, interrupt_before=[HUMAN_REVIEW])

while True:
    user_input = input("user: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    else:
        config = {"configurable": {"thread_id": "1"}}

        # 1. Run the initial user input
        for event in app.stream({"messages": [("user", user_input)]}, config=config):
            for value in event.values():
                if "messages" in value:
                    messages_update = value["messages"]
                    if isinstance(messages_update, list):
                        messages_update[-1].pretty_print()
                    else:
                        messages_update.pretty_print()

        # 2. Check if the graph is paused (waiting for human-in-the-loop)
        state = app.get_state(config)

        while state.next:
            print("\n--- GRAPH PAUSED FOR HUMAN REVIEW ---")
            feedback = input(
                "Provide feedback on the post (or type 'approve' to accept): "
            )
            if feedback.lower() in ["exit", "quit"]:
                break
            if feedback.lower() == "approve":
                # Explicitly complete the HUMAN_REVIEW node in the memory state
                app.update_state(
                    config,
                    {},  # Empty dict since we made no changes
                    as_node=HUMAN_REVIEW,
                )
                # Resume the graph to process the edge to END
                for event in app.stream(None, config=config):
                    pass
            else:
                # Add the human's feedback to the state as a new user message
                # and route it back to the POST_GENERATOR to try again
                app.update_state(
                    config,
                    {"messages": [("user", feedback)]},
                    as_node=INTENT_CLASSIFIER,  # Trick to bypass intent classifier and go back into the loop, though you may want to adjust routing
                )

                # Resume execution with the new state
                for event in app.stream(None, config=config):
                    for value in event.values():
                        if "messages" in value:
                            messages_update = value["messages"]
                            if isinstance(messages_update, list):
                                messages_update[-1].pretty_print()
                            else:
                                messages_update.pretty_print()

            # Update state to see if it paused again
            state = app.get_state(config)
