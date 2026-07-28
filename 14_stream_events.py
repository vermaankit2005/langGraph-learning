"""
Same chatbot streaming as 13_stream.py, but using astream_events.

The difference is only in the loop:
  13_stream.py  -> astream(stream_mode=["messages", "updates"])  = you label events yourself
  14_stream.py  -> astream_events()                              = LangChain labels them for you

Run:  uvicorn 14_stream_events:app --reload
Open: http://127.0.0.1:8000
"""

import json
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_tavily import TavilySearch

load_dotenv()

SERVERS = {
    "manim-server": {
        "transport": "stdio",
        "command": "C:\\Users\\verma\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
        "args": [
            "D:\\mcp-servers\\manim-mcp-server\\src\\manim_server.py"
        ],
        "env": {
            "MANIM_EXECUTABLE": "C:\\Users\\verma\\AppData\\Local\\Programs\\Python\\Python313\\Scripts\\manim.exe"
        }
    }
}

search_tool = TavilySearch(
    max_results=1,
    topic="general",
    include_images=False,
    search_depth="basic"
)


async def build_agent():
    mcp_client = MultiServerMCPClient(SERVERS)
    mcp_tools = await mcp_client.get_tools()

    print("Available tools:")
    for tool in mcp_tools:
        print(f"- {tool.name}: {tool.description}")

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.8, reasoning_format="hidden")

    return create_agent(model=llm, tools=[search_tool, *mcp_tools])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the agent ONCE when the server boots, not on every request.
    app.state.agent = await build_agent()
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------- streaming

async def event_generator(agent, question: str):
    """
    astream_events gives ONE flat stream where every item is already labelled.
    We only care about 3 labels; everything else is ignored.

        on_chat_model_stream -> the model typed a piece of text
        on_tool_start        -> a tool began running
        on_tool_end          -> that tool finished
    """
    stream = agent.astream_events(
        {"messages": [HumanMessage(content=question)]},
        version="v2",
    )

    async for event in stream:
        kind = event["event"]

        # ---- the model is typing ------------------------------------
        if kind == "on_chat_model_stream":
            piece = event["data"]["chunk"].text
            if piece:                       # skip empty chunks (tool-call planning)
                yield json.dumps({"type": "token", "text": piece})

        # ---- a tool started ------------------------------------------
        elif kind == "on_tool_start":
            yield json.dumps({
                "type": "tool_start",
                "id": event["run_id"],      # pairs with on_tool_end below
                "name": event["name"],
                "args": event["data"].get("input"),
            })

        # ---- that tool finished --------------------------------------
        elif kind == "on_tool_end":
            output = event["data"].get("output")
            yield json.dumps({
                "type": "tool_end",
                "id": event["run_id"],      # same run_id -> same box on screen
                "name": event["name"],
                "result": str(getattr(output, "text", output))[:2000],
            })

    yield json.dumps({"type": "done"})


@app.get("/stream")
async def stream_endpoint(q: str = Query("What is the current temp of new delhi")):
    # EventSourceResponse adds the "data: " prefix, the blank line, keep-alive
    # pings, and stops the generator when the browser tab is closed.
    return EventSourceResponse(event_generator(app.state.agent, q))


# ---------------------------------------------------------------- test page

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<meta charset="utf-8">
<title>astream_events demo</title>
<style>
  body { font: 16px/1.6 system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
  #chat { margin-top: 20px; }
  #chat > div { white-space: pre-wrap; }
  .tool { background: #f4f4f5; border-left: 3px solid #999; padding: 8px 12px; margin: 10px 0;
          border-radius: 4px; font-size: 14px; }
  .tool.running { border-color: #d97706; }
  .tool.done    { border-color: #16a34a; }
  input { width: 75%; padding: 8px; } button { padding: 8px 16px; }
</style>

<h2>astream_events demo</h2>
<input id="q" value="What is the current temp of new delhi">
<button onclick="ask()">Ask</button>
<div id="chat"></div>

<script>
function ask() {
  const chat = document.getElementById("chat");
  chat.innerHTML = "";

  const tools = {};            // run_id -> the box showing that tool
  let textBox = null;          // where the model's words go

  const es = new EventSource("/stream?q=" + encodeURIComponent(document.getElementById("q").value));

  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);

    if (ev.type === "token") {
      if (!textBox) { textBox = document.createElement("div"); chat.appendChild(textBox); }
      textBox.textContent += ev.text;
    }

    else if (ev.type === "tool_start") {
      const box = document.createElement("div");
      box.className = "tool running";
      box.textContent = "\\u23F3 Running " + ev.name + "(" + JSON.stringify(ev.args) + ")";
      chat.appendChild(box);
      tools[ev.id] = box;      // remember it, so tool_end can find it
      textBox = null;          // next words start a fresh bubble under the tool
    }

    else if (ev.type === "tool_end") {
      const box = tools[ev.id];
      if (box) {
        box.className = "tool done";
        box.textContent = "\\u2705 " + ev.name + " finished\\n" + ev.result;
      }
    }

    else if (ev.type === "done") {
      es.close();
    }
  };

  es.onerror = () => es.close();
}
</script>
"""