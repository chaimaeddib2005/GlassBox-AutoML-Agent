
import os, json
import google.generativeai as genai
from mcp_server import TOOLS, execute_tool

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# SYSTEM_PROMPT = """
# You are GlassBox Agent, an AI data scientist assistant powered by the GlassBox AutoML library.

# When a user asks you to build, train, or analyze a model, call the autofit_tool.
# After getting the tool result, explain the findings in plain, clear language.
# Always mention: the best model, its score, and the top features.
# If something fails, tell the user what went wrong.
# """
SYSTEM_PROMPT = """
You are GlassBox Agent, an AI data scientist assistant powered by a full AutoML system built from scratch.

You MUST behave like a data science report generator.

When a tool (AutoFit) returns a result:
- You MUST fully analyze ALL fields in the returned report
- Do NOT summarize or shorten unless explicitly asked
- Structure your response into:

1. Task Summary
2. Dataset Overview
3. Best Model + Why it won
4. Full Model Comparison Table
5. Feature Importance Analysis (all features if available)
6. Key Insights from EDA (missing values, outliers, correlations)
7. Evaluation Metrics (ALL metrics provided)
8. Final Interpretation (what this means in real-world terms)

Rules:
- Do NOT hide information from the tool output
- Do NOT only mention top-3 features — include all available ones
- If correlation exists, mention it
- If outliers exist, mention them
- Be clear, structured, and detailed like a ML report

Your goal is transparency and explainability.
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
            # result = execute_tool(tool_name, args)
            result = json.dumps(execute_tool(tool_name, args), indent=2)

            # Send result back to Gemini for a nice reply
            import google.generativeai.types as gtypes
            # final = chat.send_message(
            #     gtypes.content_types.to_part(
            #         {"function_response": {"name": tool_name, "response": result}}
            #     )
            # )

            final = chat.send_message(f"""
            Here is the complete AutoML report:

            {result}

            You MUST analyze all fields and produce a full structured ML report following the required format.
            """)
            return final.text

    # No tool call — just a regular reply
    return response.text