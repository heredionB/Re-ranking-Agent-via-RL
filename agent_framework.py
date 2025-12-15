
import asyncio
import json
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, TypedDict
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """Executes Python code via exec() and captures stdout."""
    try:
        namespace = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    return {"answer": answer, "submitted": True}


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-3-5-haiku-latest",
    verbose: bool = True,
) -> Any | None:

    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step+1}/{max_steps} ===")

        response = await client.messages.create(
            model=model,
            max_tokens=1000,
            tools=tools,
            messages=messages,
        )

        has_tool_use = False
        tool_results = []
        submitted = None

        for content in response.content:
            if content.type == "text" and verbose:
                print("Assistant:", content.text)

            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name
                tool_input = content.input
                handler = tool_handlers[tool_name]

                if verbose:
                    print(f"Using tool: {tool_name}")

                if tool_name == "python_expression":
                    result = handler(tool_input["expression"])
                else:
                    result = handler(tool_input["answer"])

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content.id,
                    "content": json.dumps(result)
                })

                if tool_name == "submit_answer":
                    submitted = result["answer"]

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if submitted is not None:
                return submitted

    return None
