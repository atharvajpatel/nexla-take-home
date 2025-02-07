import pandas as pd
from dotenv import load_dotenv
import os
from groq import Groq
import argparse
from typing import Dict, Any
import seaborn as sns
import numpy as np
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from matplotlib import pyplot as plt
import uuid

class SQLGenerator:
    def __init__(self):
        """Initialize the SQL Generator with hardcoded CSV path and API configurations."""
        # Load environment variables
        load_dotenv()
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # Hardcoded CSV path
        csv_path = "Copy of OpenSecrets.org _ FTX_Alameda Research Contributions - Direct Contributions & JFC Distributions.csv"
        
        # Load and process CSV
        self.df = pd.read_csv(csv_path)
        self.schema = self._generate_schema()
        
        # Initialize the graph
        self.graph = self._build_graph()

    def _llm_call(self, prompt: str) -> str:
        """Make a call to the Groq API."""
        completion = self.groq_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=150
        )
        return completion.choices[0].message.content

    def _generate_schema(self) -> str:
        """Generate schema from DataFrame."""
        columns = list(self.df.columns)
        datatypes = dict(self.df.dtypes)
        
        sample_schema = """
        Table Name: employees
        Columns:
        - id: INTEGER
        - name: TEXT
        - salary: FLOAT
        - department: TEXT
        """
        
        prompt_schema = (
            f"Based on the data, the columns are {columns} and the datatypes are {datatypes}. "
            f"Create a schema for the data. It should look like this: {sample_schema}"
        )
        return self._llm_call(prompt_schema)

    def _generate_sql_node(self, state: Dict[str, Any]) -> Dict[str, str]:
        """Node function for generating SQL from natural language."""
        query = state["query"]
        schema = state["schema"]
        prompt_sql = (
            "You are a SQL expert. Given the following table schema:\n"
            f"{schema}\n"
            "Convert the natural language query into a SQL query:\n"
            f"{query}"
        )
        response = self._llm_call(prompt_sql)
        return {"sql": response}

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for SQL generation."""
        graph_builder = StateGraph(dict)
        graph_builder.add_node("generate_sql", self._generate_sql_node)
        graph_builder.add_edge(START, "generate_sql")
        graph_builder.add_edge("generate_sql", END)
        
        # Create checkpointer without additional configuration
        checkpointer = MemorySaver()
        
        return graph_builder.compile(checkpointer=checkpointer)

    def generate_sql(self, query: str) -> str:
        """Generate SQL from natural language query."""
        initial_state = {
            "query": query,
            "schema": self.schema,
        }
        # Provide a configurable key (thread_id) to satisfy the checkpointer's requirement.
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = self.graph.invoke(initial_state, config=config)
        return result["sql"]

def main():
    """Main function to run the SQL generator interactively."""
    # Initialize SQL Generator
    sql_generator = SQLGenerator()

    # Interactive loop
    print("SQL Query Generator (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        query = input("\nEnter your query in natural language (type quit to exit): ")
        if query.lower() == 'quit':
            break
            
        try:
            sql = sql_generator.generate_sql(query)
            print("\nGenerated SQL:")
            print("-" * 50)
            print(sql)
            print("-" * 50)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
