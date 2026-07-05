# Reference implementation co-developed with Claude (Anthropic) for AI engineering practice.
# react-demo-with-tools.py
# ==============================================
# A ReAct (Reason + Act) Agent Demo with Tools
# ==============================================
#
# What is a ReAct Agent?
# ----------------------
# ReAct agents combine REASONING with ACTION:
# - They think about how to solve the user's request
# - They call tools when needed (search, calculate, etc.)
# - They observe results and continue reasoning until done
#
# Architecture Overview:
# 1. User asks a question
# 2. Model reasons: "What do I need to know?"
# 3. Model decides: Call tool X with input Y
# 4. Tool executes and returns result
# 5. Model uses result, continues reasoning...
# 6. Final answer is produced
#
# This demo shows how to build a simple ReAct agent that can use external tools.

import anthropic
import requests
import json
import datetime


# ==============================================
# STEP 1: Initialize the Anthropic Client
# ==============================================
# This creates the connection to Claude API
client = anthropic.Anthropic(api_key="your-key")

# To use locally, install the key securely:
# - Option A: Use environment variable: export ANTHROPIC_KEY=...
# - Option B: Create .env file: ANTHROPIC_KEY=... (gitignore this!)


# ==============================================
# STEP 2: Define Available Tools
# ==============================================
# These are the tools that Claude can see and call.
# Each tool has a name, description, and input schema.

tools = [
    {
        "name": "web_search",
        # Description explains what this tool does to Claude
        "description": "Search the web for current information. Use this when you need facts you don't know.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    # Description guides Claude on what kind of queries to make
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use this for any arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "e.g. '(42 * 1.2) + 100'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_date",
        "description": "Returns today's date. Use this when the user asks about today or current time.",
        "input_schema": {
            "type": "object",
            "properties": {},
            # No required arguments for this tool
            "required": []
        }
    }
]


# ==============================================
# STEP 3: Implement Tool Functions
# ==============================================
# These are the actual Python functions that execute when called.
# They receive inputs from Claude and return text results.

def web_search(query: str) -> str:
    """
    Performs a web search for the given query.

    Args:
        query: The search term to look up (e.g., "Python programming language")

    Returns:
        A string containing search results as plain text.

    Note:
        This is a FAKE implementation for demo purposes.
        Replace with real API call in production.
    """
    # Fake version - replace with real Tavily or Serper API call
    # Real implementation example:
    # response = requests.post(
    #     "https://api.tavily.com/search",
    #     headers={"Authorization": f"Bearer {tavily_key}"},
    #     json={"query": query}
    # )
    # return response.json()["response"]

    # Demo returns a fixed answer for demonstration
    return f"[Fake search result for '{query}']: Python was created by Guido van Rossum and released in 1991."


def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression.

    Args:
        expression: A string containing a math expression (e.g., "2 + 3 * 4")

    Returns:
        String representation of the result.

    Warning:
        Uses eval() which can be dangerous with untrusted input.
        In production, use a safe math parser or restrict allowed functions.
    """
    try:
        # Evaluate and return result as string
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_current_date() -> str:
    """
    Returns the current date and day of week.

    Returns:
        Formatted date string like "Friday, June 28, 2026"
    """
    # Returns formatted current date (e.g., "Monday, January 15, 2026")
    return datetime.datetime.now().strftime("%A, %B %d, %Y")


# ==============================================
# STEP 4: Create a Tool Dispatch Function
# ==============================================
# This function maps tool names to their implementations.
# When Claude calls a tool by name, we route it here.

def dispatch_tool(name: str, inputs: dict) -> str:
    """
    Dispatches a tool call to its implementation function.

    Args:
        name: The tool name (matches the one in tools list)
        inputs: Dictionary of arguments from Claude's tool call

    Returns:
        Result string from the tool implementation

    Example:
        dispatch_tool("web_search", {"query": "What is Python?"})
        returns the web search result text
    """
    if name == "web_search":
        return web_search(inputs["query"])
    elif name == "calculate":
        return calculate(inputs["expression"])
    elif name == "get_current_date":
        return get_current_date()
    else:
        return f"Error: unknown tool '{name}'"


# ==============================================
# STEP 5: Format Tools for Anthropic API
# ==============================================
# Convert our tools dict to the format Anthropic expects.

def format_tools(tools_list):
    """
    Converts a list of tool dicts into Anthropic's ToolChoice format.

    Args:
        tools_list: List of tool dicts with name, description, input_schema

    Returns:
        List of properly formatted Tool objects for Anthropic API
    """
    formatted_tools = []

    for tool in tools_list:
        tool_obj = {
            "name": tool["name"],
            "description": tool["description"],
            # Schema gets converted to JSON format
            "input_schema": json.dumps(tool["input_schema"])
        }
        formatted_tools.append(tool_obj)

    return formatted_tools


# ==============================================
# STEP 6: Build the Complete Chat Loop
# ==============================================
# This function handles the full ReAct interaction loop.

def chat_with_agent(prompt: str):
    """
    Engages the ReAct agent in a conversation with the user prompt.

    Args:
        prompt: The user's initial question (e.g., "What is Python?")

    Returns:
        List of message dicts representing the full conversation

    The loop works as follows:
    1. User sends prompt to Claude
    2. Claude reasons and decides whether to call a tool or answer directly
    3. If tool called: Execute it, add result back to conversation
    4. Repeat until Claude is confident in its final answer
    """
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Get tools in Anthropic format
    formatted_tools = format_tools(tools)

    print("\n" + "=" * 60)
    print("🤖 REACT AGENT DEMO")
    print("=" * 60)
    print(f"\nUser: {prompt}")

    # Continue the ReAct loop
    max_turns = 20  # Safety limit to prevent infinite loops
    turn = 0

    while turn < max_turns:
        turn += 1
        print(f"\n--- Turn {turn} ---")
        print("Thinking...\n")

        try:
            # Make the API call with streaming disabled for simplicity
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Use latest model
                max_tokens=1000,
                messages=messages,
                tools=formatted_tools,
                temperature=0.7  # Creative for reasoning tasks
            )

            # Process response
            assistant_message = response.content[0]

            if "tool_use" in assistant_message.type:
                # Tool call requested
                print(f"🛠️ Agent decided to call tool: {assistant_message.name}")
                inputs = assistant_message.input

                # Execute the tool
                result = dispatch_tool(assistant_message.name, inputs)

                print(f"📊 Tool Result:\n{result}\n")

                # Add tool result back to conversation
                messages.append(assistant_message.to_message_part())
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": assistant_message.id,
                            "content": result
                        }
                    ]
                })

            elif "text" in assistant_message.type:
                # Final answer
                print(f"✅ Agent says:\n{assistant_message.text}")
                print("\n" + "=" * 60)
                print("🎯 Conversation Complete")
                print("=" * 60 + "\n")
                break

        except Exception as e:
            print(f"\n❌ Error: {e}\n")

    return messages


# ==============================================
# STEP 7: Run the Demo
# ==============================================
if __name__ == "__main__":
    # Example interaction - user asks a question
    # Try different prompts:
    # - "What is Python?" (uses web_search)
    # - "Calculate 25 * 1.5 + 50" (uses calculate)
    # - "What day is it today?" (uses get_current_date)

    print("📋 Available Prompts:")
    print("- " + "-" * 40)
    print("Type any of these in quotes, or enter your own question:")
    print("- 'What is Python?'")
    print("- 'Calculate 25 * 1.5 + 50'")
    print("- 'What day is it today?'")
    print("- 'How many seconds in a year?'")
    print("- Your custom question here\n")

    # Get user input (can be string or interactively typed)
    # For now, let's demo with an example prompt
    demo_prompt = "What is Python?"

    # Run the chat loop
    messages = chat_with_agent(demo_prompt)
