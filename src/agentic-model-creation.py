import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent

# 1. Load your CSV data
df = pd.read_csv("../Data/train.csv")

# 2. Define the tool using the standard @tool decorator
@tool
def run_python_on_dataframe(code: str) -> str:
    """
    Execute python code on a pandas DataFrame named 'df'. 
    Use this to look at the data, get row counts, calculate averages, or plot/filter data.
    Input should be a string containing valid Python code.
    Always use print() to output results.
    """
    try:
        import sys
        from io import StringIO
        
        local_vars = {"df": df, "pd": pd}
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        
        # Run the LLM's generated python code
        exec(code, {}, local_vars)
        
        sys.stdout = old_stdout
        output = redirected_output.getvalue()
        
        if not output.strip():
            return "Code executed successfully, but returned no output. Remember to use print() to see results."
        return output
    except Exception as e:
        return f"Error executing code: {str(e)}"

tools = [run_python_on_dataframe]

# 3. Setup ChatOllama 
llm = ChatOllama(
    model="llama3.2:latest", 
    temperature=0
)

# 4. Construct the system instructions
system_prompt = f"""You are a data analysis assistant. You have access to a pandas DataFrame named `df`.
The columns in this DataFrame are: {list(df.columns)}

To answer any user question, write Python code and pass it to the `run_python_on_dataframe` tool. 
ALWAYS use `print()` inside your code snippet to output data so you can see it!
"""

# 5. Create the agent using the correct 'model' keyword argument
agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

# 6. Test your query
question = "How many rows are in the dataset and what are the first 3 rows?"
print(f"Question: {question}\n")

# Run it!
response = agent_executor.invoke({"messages": [("user", question)]})

# Print the final result message
print("\n--- Final Answer ---")
print(response["messages"][-1].content)