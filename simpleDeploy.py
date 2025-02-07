import os
from dotenv import load_dotenv
from groq import Groq

def get_sql_query(user_query: str) -> str:
    """
    Generate SQL query using Groq LLM with predefined schema.
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize Groq client
    groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    # Predefined schema
    schema = """
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
    
    # Construct prompt
    prompt = f"""You are a SQL expert. Given the following table schema:
    {schema}
    
    Convert this natural language query into a SQL query:
    {user_query}
    
    Please provide only the SQL query without any explanations."""

    # Make API call
    completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=150
    )
    
    return completion.choices[0].message.content

def main():
    print("SQL Query Generator (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        query = input("\nEnter your query in natural language (enter quit to exit): ")
        if query.lower() == 'quit':
            break
            
        try:
            sql = get_sql_query(query)
            print("\nGenerated SQL:")
            print("-" * 50)
            print(sql)
            print("-" * 50)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()