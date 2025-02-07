import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from typing import Dict, Any

# -------------------------------
# Custom Groq LLM Implementation
# -------------------------------
class GroqLLM(DeepEvalBaseLLM):
    def __init__(self, model):
        """Initialize the GroqLLM with a Groq client."""
        self.model = model

    def load_model(self):
        """Returns the underlying Groq model (client)."""
        return self.model

    def generate(self, prompt: str) -> str:
        """Synchronously generate a completion using Groq."""
        chat_model = self.load_model()
        response = chat_model.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        """Asynchronously generate a completion using Groq.
        Since Groq doesn't support async, we'll use the sync version."""
        return self.generate(prompt)

    def get_model_name(self):
        """Returns model identifier string."""
        return "Custom Groq LLM"

# -------------------------------
# SQL Query Generation
# -------------------------------
class SQLQueryGenerator:
    SCHEMA = """
    Table Name: campaign_contributions
    Columns:
    - cycle: INTEGER (election cycle year)
    - state_federal: TEXT (state or federal elections)
    - contribid: TEXT (unique contributor ID)
    - contrib: TEXT (contributor name)
    - city: TEXT (contributor city)
    - state: TEXT (contributor state)
    - zip: TEXT (zip code)
    - fecoccemp: TEXT (occupation/employer)
    - orgname: TEXT (organization name)
    - ultorg: TEXT (ultimate organization)
    - date: DATE (contribution date)
    - amount: DECIMAL (contribution amount)
    - recipid: TEXT (recipient ID)
    - recipient: TEXT (recipient name)
    - party: TEXT (political party)
    - recipcode: TEXT (recipient code)
    - type: TEXT (contribution type)
    - fectransid: TEXT (FEC transaction ID)
    - pg: INTEGER (page number)
    - cmteid: TEXT (committee ID)
    """

    def __init__(self, llm_model: GroqLLM):
        """Initialize with a GroqLLM instance."""
        self.llm = llm_model

    def generate_query(self, user_query: str) -> str:
        """Generate SQL query from natural language input."""
        prompt = f"""You are a SQL expert. Given the following table schema:
{self.SCHEMA}

Convert this natural language query into a SQL query:
{user_query}

Please provide only the SQL query without any explanations."""
        
        return self.llm.generate(prompt)

# -------------------------------
# Evaluation Handler
# -------------------------------
class QueryEvaluator:
    def __init__(self, llm_model: GroqLLM):
        """Initialize evaluator with LLM model."""
        self.llm = llm_model
        self.metric = AnswerRelevancyMetric(model=llm_model)
        self.query_generator = SQLQueryGenerator(llm_model)

    def evaluate_queries(self, ground_truth_path: str) -> pd.DataFrame:
        """Evaluate SQL query generation against ground truth data."""
        # Load ground truth data
        ground_truth_df = pd.read_csv(ground_truth_path)
        evaluation_results = []

        for idx, row in ground_truth_df.iterrows():
            question = row["question"]
            ground_truth = row["answer"]
            
            # Generate SQL query
            generated_sql = self.query_generator.generate_query(question)
            
            # Create and evaluate test case
            test_case = LLMTestCase(
                input=question,
                actual_output=generated_sql,
                expected_output=ground_truth
            )
            
            try:
                assert_test(test_case, [self.metric])
                result_str = "PASS"
            except AssertionError as ae:
                result_str = f"FAIL: {str(ae)}"
            
            print(f"Test Case {idx + 1}:")
            print(f"Question: {question}")
            print(f"Ground Truth SQL: {ground_truth}")
            print(f"Generated SQL: {generated_sql}")
            print(f"Result: {result_str}")
            print("-" * 50)
            
            evaluation_results.append({
                "question": question,
                "ground_truth": ground_truth,
                "generated_sql": generated_sql,
                "result": result_str
            })
        
        return pd.DataFrame(evaluation_results)

# -------------------------------
# Main Execution
# -------------------------------
def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Groq model
    custom_model = Groq(api_key=os.getenv('GROQ_API_KEY'))
    groq_llm = GroqLLM(model=custom_model)
    
    # Initialize evaluator
    evaluator = QueryEvaluator(groq_llm)
    
    # Run evaluation
    results_df = evaluator.evaluate_queries("ground_truth.csv")
    
    # Save results
    results_df.to_csv("deepeval_results.csv", index=False)
    print("Evaluation complete. Results saved to 'deepeval_results.csv'.")

if __name__ == "__main__":
    main()