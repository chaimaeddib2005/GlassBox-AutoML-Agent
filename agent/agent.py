
import os, json
import google.generativeai as genai
from mcp_server import TOOLS, execute_tool

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = """
You are GlassBox Agent, an AI data scientist assistant powered by the GlassBox AutoML library.

When a user asks you to build, train, or analyze a model, call the autofit_tool.
After getting the tool result, explain the findings in plain, clear language.
Always mention: the best model, its score, and the top features.
If something fails, tell the user what went wrong.
"""

def run_agent(user_message: str, csv_path: str = None) -> str:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        system_instruction=SYSTEM_PROMPT,
        tools=TOOLS
    )
    chat = model.start_chat(enable_automatic_function_calling=False)

    # Inject csv path into message if provided
    full_message = user_message
    if csv_path:
        full_message += f"\n\n[CSV file is available at: {csv_path}]"

    response = chat.send_message(full_message)

    # Check if Gemini wants to call a tool
    for part in response.parts:
        if hasattr(part, "function_call") and part.function_call:
            tool_name = part.function_call.name
            args = dict(part.function_call.args)

            # Execute tool (runs your AutoFit)
            result = execute_tool(tool_name, args)

            # Send result back to Gemini for a nice reply
            import google.generativeai.types as gtypes
            final = chat.send_message(
                gtypes.content_types.to_part(
                    {"function_response": {"name": tool_name, "response": result}}
                )
            )
            return final.text

    # No tool call — just a regular reply
    return response.text