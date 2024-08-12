# LLM-Powered Adaptive Data Analysis Agent

## Overview

This project implements an advanced, LLM-powered agent capable of performing complex data analysis tasks across multiple
data sources. Using natural language interactions, the agent can query databases, generate visualizations, and create
data-driven narratives, adapting its approach based on user intent and available data.

## Key Features

- LLM-driven reasoning for task planning and execution
- Dynamic generation of SQL and Elasticsearch queries
- Autonomous data visualization selection and creation
- AI-powered data storytelling and insight generation
- Multi-step, self-correcting workflow with explicit reasoning
- Seamless switching between data sources based on query context
- Interactive refinement of queries and outputs

## Technical Stack

- LangChain & LangGraph: Agent orchestration and reasoning
- OpenAI GPT-4: Core language model for decision-making and content generation
- FastAPI: Backend API for agent interactions
- MySQL & Elasticsearch: Supported data sources
- Langfuse: Agent tracing and performance monitoring
- React & D3.js/Plotly: Frontend for user interaction and data visualization

## Agent Architecture

1. Intent Classifier: Determines the high-level goal of the user's request
2. Query Analyzer: Distinguishes between data retrieval, visualization, and storytelling tasks
3. Task Planner: Breaks down the goal into a series of actionable steps
4. Context Manager: Maintains and updates the agent's understanding of the current state
5. Schema Retriever: Fetches database schemas and Elasticsearch mappings
6. Query Generator: Creates SQL and Elasticsearch queries based on user intent
7. Query Validator: Ensures generated queries are valid and safe to execute
8. Tool Selector: Chooses appropriate tools (e.g., SQL query, visualization) for each step
9. Execution Engine: Runs selected tools and processes their outputs
10. Reasoning Engine: Evaluates results, makes decisions, and plans next steps
11. Output Generator: Formulates human-readable responses and visualizations
12. Visualization Generator: Creates appropriate data visualizations
13. Storyline Creator: Generates narrative insights from data analysis

## Example Interaction

User: "Analyze our Q2 sales performance and visualize the top-performing products."

Agent:

1. Classifies intent as a multi-step analysis task
2. Analyzes query to determine data retrieval and visualization needs
3. Plans steps: retrieve Q2 sales data, identify top products, generate visualization
4. Retrieves relevant database schema
5. Generates and validates SQL query to fetch Q2 sales data
6. Executes query and processes results to identify top-performing products
7. Selects and generates appropriate visualization (e.g., bar chart)
8. Creates a data-driven narrative summarizing key insights
9. Presents visualization, summary, and storyline to user

## Setup and Usage

[Setup instructions here]

## Extending the Agent

To add new capabilities:

1. Implement a new Tool class (e.g., NewDataSourceTool, AdvancedVisualizationTool)
2. Update the Tool Selector to consider the new tool
3. Enhance the Task Planner and Query Analyzer to incorporate the new capability
4. Add relevant prompts and few-shot examples for the LLM
5. Extend the Schema Retriever and Query Generator if adding a new data source
6. Update the Visualization Generator for new chart types or data representations
7. Enhance the Storyline Creator to incorporate new types of insights

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.