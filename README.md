# Natural Language to SQL Query Converter

This project implements a natural language to SQL query converter using LLMs and evaluation frameworks.

## Setup and Installation

1. Clone the repository
```bash
# Navigate to the root repository
cd Documents/Github/nexla-take-home
```

2. Create and activate virtual environment
```bash
# Create virtual environment
python -m venv nexenv

# Activate virtual environment
# On Windows:
nexenv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

The project includes multiple interfaces for natural language to SQL conversion:

### Simple Deployment
```bash
python simpleDeploy.py
```
This provides a command-line interface for basic query conversion. Type 'quit' to exit.

### Advanced Deployment
```bash
python deploy.py
```
This version uses a Langgraph agent framework for more flexible query processing.

### Evaluation
```bash
# First, ensure no existing results file
rm deepeval_results.csv

# Run evaluation
python evaluation.py
```
Results will be saved in `deepeval_results.csv`, which can be analyzed using pandas for better visibility.

## Technical Stack

- **LLM Models**: 
  - Groq (llama-3.3-70b)
  - Claude 3.5 Sonnet
  - O3-mini-high

- **Frameworks & Libraries**:
  - Langchain/Langgraph
  - Deepeval
  - Pandas

## Development Process

### Initial Approach (simpleDeploy.py)
- Implemented straightforward schema-as-context approach
- Direct pass-through to Groq model
- Infinite chatbot functionality
- Attempted code extraction using regex/pydantic (unsuccessful)

### Enhanced Approach (deploy.py)
- Built modular Langgraph agent framework
- Dynamic schema creation from column names and datatypes
- Flexible architecture allowing for future agent additions
- CSV compatibility

### Evaluation Framework
- Created ground truth dataset using Claude 3.5 Sonnet
- Implemented Deepeval framework for testing
- Evaluated simpleDeploy flow against ground truth

## Strengths & Limitations

### Strengths
- Robust evaluation framework using Deepeval
- First-time implementation of ground truth evaluation
- Demonstrated quick learning and adaptation

### Limitations
- No actual SQL execution capability
- Challenges in consistent LLM output parsing
- Limited SQL database integration

## Future Improvements

With additional time, potential improvements include:

- Implementation of consistent code output parsing
- Full SQL database integration
- Query execution capabilities
- Extended evaluation framework for Langgraph approach

