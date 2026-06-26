from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph, MessagesState

load_dotenv()  # reads GROQ_API_KEY from your .env

SYSTEM_PROMPT = "You are a concise, friendly assistant. Keep replies short."

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.7, reasoning_format="hidden")


def chat_node(state: MessagesState) -> MessagesState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    reply = llm.invoke(messages)
    return {"messages": [reply]}  # appended to state by the add_messages reducer


# 3) GRAPH -------------------------------------------------------------------
graph = StateGraph(MessagesState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# 4) CHECKPOINTER ------------------------------------------------------------
DB_URI = "postgresql://postgres:postgres@localhost:5432/langgraph"


def main():
    # Try connection pool here.
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "demo-conversation-2"}}

        print("LangGraph bot ready (in-memory checkpointer). Type 'bye' to quit.")
        print("Tip: it remembers earlier turns because every call uses the SAME thread_id.\n")

        while True:
            user_text = input("You: ")
            if user_text.lower().strip() in ("bye", "quit", "exit"):
                break

            # We only send the NEW message. The checkpointer supplies the history.
            result = app.invoke(
                {"messages": [HumanMessage(content=user_text)]},
                config=config,
            )
            print("Bot:", result["messages"][-1].content, "\n")

        # Proof it was all kept in state: print the full remembered transcript.
        final = app.get_state(config).values["messages"]
        print(f"\n--- Conversation had {len(final)} messages stored in memory --- \n")
        for i, msg in enumerate(final, 1):
            print(f"{i}: {msg.type.upper()}: {msg.content}")


if __name__ == "__main__":
    main()
