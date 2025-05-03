# groq_agent.py

from langchain.agents import initialize_agent, Tool
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import re

# ===== Define Tools =====

def calculator_tool(query: str):
    try:
        result = eval(query)
        return f"The result is {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def date_tool(query: str):
    today = datetime.now().date()
    if "today" in query.lower():
        return f"Today's date is {today}"

    dates = re.findall(r"\d{4}-\d{2}-\d{2}", query)
    if len(dates) == 2:
        d1 = datetime.strptime(dates[0], "%Y-%m-%d")
        d2 = datetime.strptime(dates[1], "%Y-%m-%d")
        diff = abs((d2 - d1).days)
        return f"The difference is {diff} days."

    return "Please provide two dates in YYYY-MM-DD format."


def user_info(query: str):
    today = datetime.now().date()
    if "today" in query.lower():
        return f"Today's date is {today}"

    dates = re.findall(r"\d{4}-\d{2}-\d{2}", query)
    if len(dates) == 2:
        d1 = datetime.strptime(dates[0], "%Y-%m-%d")
        d2 = datetime.strptime(dates[1], "%Y-%m-%d")
        diff = abs((d2 - d1).days)
        return f"The difference is {diff} days."

    return "Please provide two dates in YYYY-MM-DD format."



# ===== Tool List =====
tools = [
    Tool(name="Calculator", func=calculator_tool, description="Handles math like 2 + 2, 4 * 5."),
    Tool(name="DateTool", func=date_tool, description="Tells today's date or difference between two dates.")
]

# ===== Memory =====
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ===== Groq LLM and Agent =====
llm = ChatGroq(model_name="llama-3.1-8b-instant",temperature=0)  # Or "llama3-70b-8192" if available
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", memory=memory, verbose=True, max_iterations=1)

# ===== Run Agent =====
def run_agent():
    # print(agent.run("What is 8 * 12?"))
    # print(agent.run("What is today's date?"))
    print(agent.run("How many days between 2023-11-01 and 2024-01-01?"))

if __name__ == "__main__":
    run_agent()
