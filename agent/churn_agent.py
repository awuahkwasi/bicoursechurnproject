import json
import os
import sys
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent))
from tools import TOOL_REGISTRY  # noqa: E402

load_dotenv()

MODEL_NAME = "qwen/qwen3.6-27b"

SYSTEM_PROMPT = """You are a Customer Churn Analyst AI Agent for a telecom company.

You help business stakeholders understand which customers are at risk of
churning, why, and what to do about it. You have access to tools backed by
a trained machine learning model and the company's live customer dataset.

Guidelines:
- Always use the tools to get real numbers. Never invent statistics.
- When asked about a specific customer, use predict_churn_for_customer.
- When asked "who is most at risk", use get_top_risk_customers.
- When asked about trends across a category (contract type, payment method,
  internet service, etc.), use get_churn_rate_by_segment.
- When asked "what if" about a hypothetical customer profile, use
  predict_churn_for_new_profile.
- After calling tools, explain the result in plain business language and,
  where relevant, suggest a concrete retention action (e.g. offer a longer
  contract, waive a fee, proactive outreach).
- Be concise. Use bullet points for lists of customers or numbers.
- Always pass all tool parameters explicitly, even optional ones.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "predict_churn_for_customer",
            "description": "Look up an existing customer by their customer_id and return their churn risk and key attributes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The customer ID, e.g. CUST-00001",
                    },
                },
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_churn_summary_stats",
            "description": "Get overall churn statistics across the entire customer base (totals, risk tier counts, averages). No parameters needed.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_risk_customers",
            "description": "Get the N customers with the highest predicted churn probability, ranked descending. Always pass top_n explicitly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "How many customers to return. Must be provided explicitly, e.g. 5 or 10.",
                    },
                },
                "required": ["top_n"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_churn_rate_by_segment",
            "description": (
                "Get the churn rate (%) broken down by a categorical column such as "
                "contract_type, payment_method, internet_service, location, gender, "
                "education_level, or marital_status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to group by, e.g. contract_type",
                    },
                },
                "required": ["column"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_churn_for_new_profile",
            "description": (
                "Predict churn probability for a hypothetical customer that does not "
                "exist yet, given a dict of feature values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "object",
                        "description": (
                            "Dict of feature_name: value pairs, e.g. "
                            "{\"tenure_months\": 3, \"contract_type\": \"Month-to-month\", "
                            "\"monthly_charges\": 75.0, \"customer_service_calls\": 4}"
                        ),
                    },
                },
                "required": ["profile"],
                "additionalProperties": False,
            },
        },
    },
]


class ChurnAgent:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("No Groq API key found. Set GROQ_API_KEY in your .env file.")
        self.client = Groq(api_key=api_key)
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool '{tool_name}'"})
        try:
            result = fn(**tool_input)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def ask(self, user_message: str, max_tool_rounds: int = 5) -> str:
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(max_tool_rounds):
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                max_tokens=1500,
            )

            message = response.choices[0].message
            # Append as dict so it's always serialisable
            assistant_msg = {
                "role": "assistant",
                "content": message.content,
            }
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            self.messages.append(assistant_msg)

            if not message.tool_calls:
                return message.content or ""

            for call in message.tool_calls:
                try:
                    tool_input = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                result = self._execute_tool(call.function.name, tool_input)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })

        return "I wasn't able to finish that request within the allowed number of tool calls."

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def main():
    print("Customer Churn AI Agent (Groq) — type 'exit' to quit, 'reset' to start over\n")
    agent = ChurnAgent()
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("Conversation reset.\n")
            continue

        try:
            answer = agent.ask(user_input)
            print(f"\nAgent: {answer}\n")
        except Exception as e:
            print(f"\n[Error] {e}\n")


if __name__ == "__main__":
    main()