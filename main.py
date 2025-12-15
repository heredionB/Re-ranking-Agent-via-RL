
import os
os.environ["ANTHROPIC_API_KEY"] = "api_key"
#from anthropic import AsyncAnthropic
import asyncio
from tasks.rearank_task import PROMPT
from tasks.rearank_grader import grade_answer
from agent_framework import (
    python_expression_tool,
    submit_answer_tool,
    run_agent_loop,
)


async def main():
    tools = [
        {
            "name": "python_expression",
            "description": "Executes Python code",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {}},
                "required": ["answer"],
            },
        },
    ]

    handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    print("\nRunning REARANK RL Task...\n")

    result = await run_agent_loop(
        PROMPT,
        tools,
        handlers,
        max_steps=10,
        verbose=True,
    )

    print("\nMODEL OUTPUT:", result)
    print("PASS?", grade_answer(result))


if __name__ == "__main__":
    asyncio.run(main())
