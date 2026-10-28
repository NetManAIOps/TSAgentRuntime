"""Harness agent: LLM loop over GymTS LibraryTSTools.

Two modes:
  CLI:      python harness.py ["task text"]        # one-shot, prints JSON result
  HTTP:     PORT=9000 python harness.py --serve    # AgentRuntime container mode

HTTP endpoints (AgentRuntime container contract):
  GET  /health                  -> {"status": "ok"}
  POST /v1/chat/completions     -> OpenAI-compatible non-streaming chat
  POST /compatible-mode/v1/responses -> Responses API JSON or SSE stream
  POST /chat                    -> {"message": str} -> {"reply": result}

Every model exchange and tool call is journaled to JSONL and exported as
OpenTelemetry GenAI spans; LibraryTSTools persists each native call as CAS evidence.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from gymts import GymTS
from gymts_libraryts.tools import LibraryTSTools
from opentelemetry.util.genai.extended_handler import get_extended_telemetry_handler
from opentelemetry.util.genai.extended_types import (
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)

API_BASE = os.environ.get("LLM_BASE_URL", "https://kuaipao.pro/v1")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4.1-flash-1017")

SKTIME_REPO = os.environ.get("SKTIME_REPO", "/tmp/sktime-fixed")
SKTIME_REVISION = os.environ.get("SKTIME_REVISION", "080f4c3c3695f239e328548bac7d7a3ef7aa9127")
GYM_STORE = os.environ.get("GYM_STORE", "/tmp/gymts-state")
JOURNAL = Path(os.environ.get("HARNESS_JOURNAL", "/tmp/gymts-harness/journal.jsonl"))
INDEX_PAGE = Path(os.environ.get("HARNESS_INDEX", "/opt/harness/index.html"))

def schema(description, arguments, params=None):
    properties = {"arguments": arguments}
    if params is not None:
        properties["params"] = params
    return {
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["arguments"],
            "additionalProperties": False,
        },
    }


def array(items=None):
    return {"type": "array", "items": items or {"type": "number"}}


OBJECT = {"type": "object", "additionalProperties": True}
TOOL_SCHEMAS = {
    "dataset.load": schema(
        "加载 LibraryTS 内置数据集，返回完整序列。支持 airline、lynx、shampoo-sales。",
        {"type": "object", "properties": {"dataset": {"type": "string", "enum": ["airline", "lynx", "shampoo-sales"]}}, "required": ["dataset"], "additionalProperties": False},
        {"type": "object", "additionalProperties": False},
    ),
    "forecast.naive": schema(
        "使用真实 NaiveForecaster 对给定数值序列预测；horizons 是从 1 开始的相对步数。",
        {"type": "object", "properties": {"values": array(), "horizons": array({"type": "integer", "minimum": 1})}, "required": ["values", "horizons"], "additionalProperties": False},
        {"type": "object", "properties": {"strategy": {"type": "string", "enum": ["last", "mean", "drift"]}}, "additionalProperties": False},
    ),
    "forecast.polynomial": schema(
        "使用真实 PolynomialTrendForecaster 对给定数值序列做多项式趋势预测。",
        {"type": "object", "properties": {"values": array(), "horizons": array({"type": "integer", "minimum": 1})}, "required": ["values", "horizons"], "additionalProperties": False},
        {"type": "object", "properties": {"degree": {"type": "integer", "enum": [1, 2, 3]}}, "additionalProperties": False},
    ),
    "detect.threshold": schema(
        "使用 ThresholdDetector 检测给定序列中的阈值异常区段。",
        {"type": "object", "properties": {"values": array()}, "required": ["values"], "additionalProperties": False},
        {"type": "object", "properties": {"upper": {"type": "number"}, "lower": {"type": ["string", "null"], "enum": ["-upper", None]}}, "additionalProperties": False},
    ),
    "analyze.seasonality": schema(
        "使用 ACF 或 periodogram 分析给定数值序列的季节周期。",
        {"type": "object", "properties": {"values": array()}, "required": ["values"], "additionalProperties": False},
        {"type": "object", "properties": {"method": {"type": "string", "enum": ["acf", "periodogram"]}}, "additionalProperties": True},
    ),
    "distance.euclidean": schema(
        "计算两个等长数值序列的真实欧氏距离。",
        {"type": "object", "properties": {"x": array(), "y": array()}, "required": ["x", "y"], "additionalProperties": False},
        OBJECT,
    ),
    "labts.catalog": schema(
        "查看 LabTS 可用算法目录；运行 labts.run 前可用它查询 algorithm_id。",
        {"type": "object", "properties": {"include_registered": {"type": "boolean"}}, "additionalProperties": False},
        {"type": "object", "additionalProperties": False},
    ),
    "labts.run": schema(
        "运行一个独立 LabTS 库实验。通过 arguments.params 或 params 传实验参数，例如 horizon。",
        {"type": "object", "properties": {"task": {"type": "string"}, "dataset_id": {"type": "string"}, "algorithm_id": {"type": "string"}, "params": OBJECT}, "required": ["task", "dataset_id", "algorithm_id"], "additionalProperties": True},
        OBJECT,
    ),
    "labts.analyze": schema(
        "运行 LabTS 原生分析算法，例如 seasonality-acf。",
        {"type": "object", "properties": {"algorithm_id": {"type": "string"}, "dataset_id": {"type": "string"}, "params": OBJECT}, "required": ["algorithm_id", "dataset_id"], "additionalProperties": True},
        OBJECT,
    ),
    "labts.metrics": schema(
        "使用 LabTS 原生指标实现计算 MAE、RMSE 等；context 提供真实值和预测值等字段。",
        {"type": "object", "properties": {"task": {"type": "string"}, "metrics": array({"type": "string"}), "context": OBJECT}, "required": ["task", "metrics", "context"], "additionalProperties": False},
        {"type": "object", "additionalProperties": False},
    ),
}
FUNCTION_TO_TOOL = {name.replace(".", "__"): name for name in TOOL_SCHEMAS}
OPENAI_TOOLS = [
    {"type": "function", "function": {"name": function_name, **TOOL_SCHEMAS[tool_name]}}
    for function_name, tool_name in FUNCTION_TO_TOOL.items()
]

SYSTEM_PROMPT = """你是一个支持普通对话和时序分析的 AI Agent。直接回答聊天、解释和知识问题；只有需要真实时序计算、数据加载、目录查询、实验或指标计算时才调用工具。不要为普通问候调用工具。

你可以自主使用 LibraryTS 的十个工具，并在一个任务中连续调用多个工具。内置数据集先用 dataset.load，再把返回的 values 原样传给计算工具。预测、异常检测、季节性、距离和指标数值必须来自真实工具结果，不得心算或编造。用户没有指定数据集或预测步数时，不要默认改写用户任务；可根据上下文选择合理值并明确说明。labts.run 是独立库实验，不会自动创建 GymTS Run 或独立评价报告。工具失败后解释原因，必要时修正参数重试。最终用自然语言直接回答用户，并列出关键结果；不要输出内部协议或 Journal。"""

DEFAULT_TASK = "请介绍你能完成哪些任务。"


def chat(messages: list[dict], max_tokens: int = 8000) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "tools": OPENAI_TOOLS,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }).encode()
    for attempt in range(3):
        req = urllib.request.Request(
            API_BASE.rstrip("/") + "/chat/completions", data=body,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 429, 500, 502, 503, 504) or attempt == 2:
                raise
            error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                json.JSONDecodeError) as exc:
            if attempt == 2:
                raise
            error = exc
        print(f"[harness] model request failed, retry {attempt + 1}/3: "
              f"{type(error).__name__}: {error}", flush=True)
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def tool_feedback(output):
    if isinstance(output, dict):
        return {key: value for key, value in output.items() if key != "provenance"}
    return output


def telemetry_messages(messages: list[dict]) -> list[InputMessage]:
    traced = []
    for message in messages:
        role = message.get("role", "user")
        if role == "system":
            continue
        parts = []
        if message.get("content"):
            parts.append(Text(content=message["content"]))
        for native_call in message.get("tool_calls") or []:
            function = native_call.get("function") or {}
            parts.append(ToolCall(
                name=function.get("name", ""),
                arguments=function.get("arguments", ""),
                id=native_call.get("id"),
            ))
        if role == "tool":
            parts.append(ToolCallResponse(
                response=message.get("content", ""),
                id=message.get("tool_call_id"),
            ))
        traced.append(InputMessage(role=role, parts=parts))
    return traced


def telemetry_output(message: dict, finish_reason: str) -> OutputMessage:
    parts = []
    if message.get("content"):
        parts.append(Text(content=message["content"]))
    if message.get("reasoning_content"):
        parts.append(Text(content=message["reasoning_content"]))
    for native_call in message.get("tool_calls") or []:
        function = native_call.get("function") or {}
        parts.append(ToolCall(
            name=function.get("name", ""),
            arguments=function.get("arguments", ""),
            id=native_call.get("id"),
        ))
    return OutputMessage(role="assistant", parts=parts, finish_reason=finish_reason)


def run_task(task: str, journal_path: Path = JOURNAL) -> dict:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    implementation = {
        "repository": SKTIME_REPO,
        "revision": SKTIME_REVISION,
        "python": sys.executable,
    }
    calls = []
    task_result = {"status": "error", "error": "model did not finish in 16 rounds",
                   "tool_calls": calls, "journal": str(journal_path)}
    handler = get_extended_telemetry_handler()
    endpoint = urllib.parse.urlsplit(API_BASE)
    entry = EntryInvocation(
        session_id=str(uuid.uuid4()),
        user_id="anonymous",
        input_messages=[InputMessage(role="user", parts=[Text(content=task)])],
    )
    agent = InvokeAgentInvocation(
        provider="deepseek",
        agent_name="GymTS LibraryTS Harness Agent",
        agent_description="Conversational time-series agent with ten LibraryTS tools",
        request_model=MODEL,
        conversation_id=entry.session_id,
        input_messages=entry.input_messages,
        system_instruction=[Text(content=SYSTEM_PROMPT)],
        tool_definitions=[
            FunctionToolDefinition(
                name=item["function"]["name"],
                description=item["function"]["description"],
                parameters=item["function"]["parameters"],
            )
            for item in OPENAI_TOOLS
        ],
        server_address=endpoint.hostname,
        server_port=endpoint.port,
    )
    with handler.entry(entry):
        with handler.invoke_agent(agent):
            total_input_tokens = 0
            total_output_tokens = 0
            with journal_path.open("a", buffering=1) as journal:
                with GymTS(GYM_STORE, integrations=("libraryts",)) as gym:
                    tools = LibraryTSTools(gym.artifacts, implementation)
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": task},
                    ]
                    for round_no in range(1, 17):
                        step = ReactStepInvocation(round=round_no)
                        with handler.react_step(step):
                            started = time.time()
                            llm = LLMInvocation(
                                provider="deepseek",
                                request_model=MODEL,
                                input_messages=telemetry_messages(messages),
                                system_instruction=[Text(content=SYSTEM_PROMPT)],
                                tool_definitions=agent.tool_definitions,
                                conversation_id=entry.session_id,
                                temperature=0.2,
                                max_tokens=8000,
                                server_address=endpoint.hostname,
                                server_port=endpoint.port,
                            )
                            with handler.llm(llm):
                                resp = chat(messages)
                                choice = resp["choices"][0]
                                msg = choice["message"]
                                content = (msg.get("content") or "").strip()
                                native_calls = msg.get("tool_calls") or []
                                usage = resp.get("usage") or {}
                                input_tokens = usage.get("prompt_tokens") or 0
                                output_tokens = usage.get("completion_tokens") or 0
                                total_input_tokens += input_tokens
                                total_output_tokens += output_tokens
                                llm.response_id = resp.get("id")
                                llm.response_model_name = resp.get("model") or MODEL
                                llm.finish_reasons = [choice.get("finish_reason") or "stop"]
                                llm.input_tokens = input_tokens
                                llm.output_tokens = output_tokens
                                llm.output_messages = [telemetry_output(
                                    msg, choice.get("finish_reason") or "stop",
                                )]
                            journal.write(json.dumps({
                                "round": round_no, "model": MODEL, "response_content": content,
                                "tool_calls": native_calls, "reasoning_content": msg.get("reasoning_content"),
                                "usage": resp.get("usage"), "seconds": round(time.time() - started, 2),
                            }, ensure_ascii=False) + "\n")
                            if not native_calls:
                                step.finish_reason = "stop"
                                if not content:
                                    task_result = {"status": "error", "error": "model returned an empty response",
                                                   "journal": str(journal_path)}
                                else:
                                    task_result = {"status": "ok", "reply": content, "tool_calls": calls,
                                                   "journal": str(journal_path)}
                                break

                            step.finish_reason = "continue"
                            assistant = {"role": "assistant", "content": msg.get("content"),
                                         "tool_calls": native_calls}
                            messages.append(assistant)
                            for native_call in native_calls:
                                function = native_call.get("function") or {}
                                tool = FUNCTION_TO_TOOL.get(function.get("name"))
                                call_id = native_call.get("id") or f"call_{round_no}_{len(calls)}"
                                tool_invocation = ExecuteToolInvocation(
                                    tool_name=tool or function.get("name") or "unknown",
                                    tool_call_id=call_id,
                                    tool_call_arguments=function.get("arguments") or "{}",
                                    tool_type="function",
                                )
                                try:
                                    with handler.execute_tool(tool_invocation):
                                        payload = json.loads(function.get("arguments") or "{}")
                                        if not isinstance(payload, dict):
                                            raise ValueError("tool arguments must be a JSON object")
                                        if tool is None:
                                            raise ValueError("unknown tool")
                                        arguments = payload.get("arguments") or {}
                                        params = payload.get("params") or {}
                                        if not isinstance(arguments, dict) or not isinstance(params, dict):
                                            raise ValueError("arguments and params must be JSON objects")
                                        call = tools.call(tool, arguments, params, parent_call_id=call_id)
                                        tool_result = {"status": "succeeded", "output": tool_feedback(call["output"])}
                                        tool_invocation.tool_call_result = tool_result
                                        calls.append({"tool": tool, "call_id": call_id, "status": "succeeded",
                                                      "call_ref": call["call_ref"]})
                                except Exception as exc:
                                    tool_result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:1000]}
                                    calls.append({"tool": tool or function.get("name"), "call_id": call_id,
                                                  "status": "failed"})
                                journal.write(json.dumps({
                                    "round": round_no, "call_id": call_id, "tool": tool,
                                    "arguments": function.get("arguments"), "result": tool_result,
                                }, ensure_ascii=False, default=str) + "\n")
                                messages.append({"role": "tool", "tool_call_id": call_id,
                                                 "content": json.dumps(tool_result, ensure_ascii=False)})
            agent.input_tokens = total_input_tokens
            agent.output_tokens = total_output_tokens
            output = task_result.get("reply") or task_result.get("error") or "Agent task failed"
            finish_reason = "stop" if task_result["status"] == "ok" else "error"
            agent.finish_reasons = [finish_reason]
            agent.output_messages = [OutputMessage(
                role="assistant", parts=[Text(content=output)], finish_reason=finish_reason,
            )]
            entry.output_messages = agent.output_messages
    task_result["trace_id"] = f"{entry.span.get_span_context().trace_id:032x}"
    return task_result


def responses_input(request: dict) -> str:
    value = request.get("input") or DEFAULT_TASK
    if isinstance(value, str):
        return value
    items = value if isinstance(value, list) else [value]
    for item in reversed(items):
        if isinstance(item, str) and item:
            return item
        if not isinstance(item, dict):
            continue
        content = item.get("content") or item.get("text") or ""
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") in ("input_text", "text")
            )
            if text:
                return text
    return json.dumps(value, ensure_ascii=False)


def response_document(response_id: str, model: str, status: str,
                      output_text: str = "", error: dict | None = None) -> dict:
    output = []
    if output_text:
        output.append({
            "id": f"msg_{response_id.removeprefix('resp_')}",
            "type": "message", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": output_text, "annotations": []}],
        })
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output,
        "output_text": output_text,
        "error": error,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


def execute_response(request: dict, response_id: str, model: str) -> dict:
    try:
        result = run_task(responses_input(request))
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    text = result.get("reply", "")
    if result["status"] == "ok":
        return response_document(response_id, model, "completed", text)
    return response_document(
        response_id, model, "failed", text,
        {"code": "agent_error", "message": result.get("error", "Agent task failed")},
    )


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_responses(self, request: dict) -> None:
        response_id = f"resp_gymts_{time.time_ns()}"
        model = request.get("model") or MODEL
        if not request.get("stream", False):
            self._send(200, execute_response(request, response_id, model))
            return

        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        sequence = 0

        def emit(event_type: str, payload: dict) -> None:
            nonlocal sequence
            sequence += 1
            payload = {"type": event_type, "sequence_number": sequence, **payload}
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.wfile.write(f"event: {event_type}\n".encode())
            self.wfile.write(b"data: " + body + b"\n\n")
            self.wfile.flush()

        try:
            emit("response.created", {
                "response": response_document(response_id, model, "in_progress"),
            })
            document = execute_response(request, response_id, model)
            if document["status"] != "completed":
                emit("response.failed", {"response": document})
                return

            text = document["output_text"]
            item = document["output"][0]
            item_id = item["id"]
            emit("response.output_item.added", {
                "output_index": 0,
                "item": {**item, "status": "in_progress", "content": []},
            })
            emit("response.content_part.added", {
                "item_id": item_id, "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            })
            for offset in range(0, len(text), 96):
                emit("response.output_text.delta", {
                    "item_id": item_id, "output_index": 0, "content_index": 0,
                    "delta": text[offset:offset + 96],
                })
            emit("response.output_text.done", {
                "item_id": item_id, "output_index": 0, "content_index": 0, "text": text,
            })
            emit("response.content_part.done", {
                "item_id": item_id, "output_index": 0, "content_index": 0,
                "part": item["content"][0],
            })
            emit("response.output_item.done", {"output_index": 0, "item": item})
            emit("response.completed", {"response": document})
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, fmt, *args):  # keep container logs readable
        print(f"[harness] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self):
        if self.path == "/":
            try:
                body = INDEX_PAGE.read_bytes()
            except OSError:
                self._send(404, {"error": "web page not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.rstrip("/") in ("/health", "/healthz"):
            self._send(200, {"status": "ok", "model": MODEL})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send(400, {"error": "invalid JSON body"})
            return

        if self.path == "/chat":
            task = request.get("message") or request.get("task") or DEFAULT_TASK
            try:
                result = run_task(task)
            except Exception as exc:
                result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            payload = dict(result)
            payload["reply"] = result.get("reply", result.get("error"))
            self._send(200 if result["status"] == "ok" else 500, payload)
            return

        if self.path.rstrip("/") == "/compatible-mode/v1/responses":
            self._send_responses(request)
            return

        if self.path.rstrip("/") == "/v1/chat/completions":
            messages = request.get("messages") or []
            task = next((m["content"] for m in reversed(messages) if m.get("role") == "user"),
                        DEFAULT_TASK)
            if not isinstance(task, str):
                task = json.dumps(task, ensure_ascii=False)
            try:
                result = run_task(task)
                content = result.get("reply") or result.get("error") or "任务失败"
            except Exception as exc:
                content = f"任务失败：{type(exc).__name__}: {exc}"
            self._send(200, {
                "id": f"chatcmpl-gymts-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model") or MODEL,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
            return

        self._send(404, {"error": "not found"})


def serve() -> int:
    port = int(os.environ.get("PORT", "9000"))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[harness] serving on {host}:{port} model={MODEL}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
    return 0


def main() -> int:
    if "--serve" in sys.argv:
        return serve()
    task = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or DEFAULT_TASK
    print(json.dumps(run_task(task), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
