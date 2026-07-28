import json
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
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


# ---------------------------------------------------------------- helpers

def sse(event_type: str, **data) -> str:
    """Format one Server-Sent Event line that the browser can read."""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


def text_of(message) -> str:
    """Get plain text out of a message whose content may be str or a list of blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    # parts = []
    # for block in content or []:
    #     if isinstance(block, str):
    #         parts.append(block)
    #     elif isinstance(block, dict) and block.get("type") == "text":
    #         parts.append(block.get("text", ""))
    # return "".join(parts)
    return ""


# ---------------------------------------------------------------- streaming

async def event_generator(agent, question: str):
    stream = agent.astream(
        input={"messages": [HumanMessage(content=question)]},
        # "messages" -> token-by-token text
        # "updates"  -> what each node produced (tool calls + tool results)
        stream_mode=["messages", "updates"],
    )

    async for mode, payload in stream:

        if mode == "messages":
            chunk, metadata = payload
            # Only the model's own words. Tool output also arrives here; skip it.
            if isinstance(chunk, AIMessageChunk):
                piece = text_of(chunk)
                if piece:
                    yield sse("token", text=piece)

        elif mode == "updates":
            for node_name, update in payload.items():
                if not isinstance(update, dict):
                    continue
                for message in update.get("messages", []):

                    # The model decided to call a tool -> tell the UI "running..."
                    if isinstance(message, AIMessage) and message.tool_calls:
                        for call in message.tool_calls:
                            yield sse(
                                "tool_start",
                                id=call["id"],
                                name=call["name"],
                                args=call["args"],
                            )

                    # The tool finished -> tell the UI the result
                    elif isinstance(message, ToolMessage):
                        yield sse(
                            "tool_end",
                            id=message.tool_call_id,
                            name=message.name,
                            status=getattr(message, "status", "success"),
                            result=text_of(message)[:200],
                        )

    yield sse("done")


@app.get("/stream")
async def stream_endpoint(q: str = Query("What is the current temp of new delhi")):
    return StreamingResponse(
        event_generator(app.state.agent, q),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stop nginx/proxies from buffering
        },
    )


# ---------------------------------------------------------------- test page

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<meta charset="utf-8">
<title>Agent stream demo</title>
<style>
  body { font: 16px/1.6 system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
  #chat { white-space: pre-wrap; margin-top: 20px; }
  .tool { background: #f4f4f5; border-left: 3px solid #999; padding: 8px 12px; margin: 10px 0;
          border-radius: 4px; font-size: 14px; white-space: pre-wrap; }
  .tool.running { border-color: #d97706; }
  .tool.done    { border-color: #16a34a; }
  input { width: 75%; padding: 8px; } button { padding: 8px 16px; }
</style>

<h2>Agent stream demo</h2>
<input id="q" value="What is the current temp of new delhi">
<button onclick="ask()">Ask</button>
<div id="chat"></div>

<script>
function ask() {
  const chat = document.getElementById("chat");
  chat.innerHTML = "";

  const tools = {};                 // tool_call id -> its box on screen
  let textBox = null;               // where model tokens go

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
      tools[ev.id] = box;
      textBox = null;               // next tokens start a fresh bubble below the tool
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
