# LLM-Powered Adaptive Data Analysis Agent

## Overview
This project implements an advanced, LLM-powered agent capable of performing complex data analysis tasks across multiple data sources. Using natural language interactions, the agent can query databases, generate visualizations, and create data-driven narratives, adapting its approach based on user intent and available data.

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
2. Task Planner: Breaks down the goal into a series of actionable steps
3. Context Manager: Maintains and updates the agent's understanding of the current state
4. Tool Selector: Chooses appropriate tools (e.g., SQL query, visualization) for each step
5. Execution Engine: Runs selected tools and processes their outputs
6. Reasoning Engine: Evaluates results, makes decisions, and plans next steps
7. Output Generator: Formulates human-readable responses and visualizations

## Example Interaction
User: "Analyze our Q2 sales performance and visualize the top-performing products."

Agent:
1. Classifies intent as a multi-step analysis task
2. Plans steps: retrieve Q2 sales data, identify top products, generate visualization
3. Selects SQL query tool to fetch Q2 sales data
4. Processes results to identify top-performing products
5. Chooses bar chart for visualization
6. Generates natural language summary of findings
7. Presents visualization and summary to user

## Setup and Usage
[Setup instructions here]

## Extending the Agent
To add new capabilities:
1. Implement a new Tool class (e.g., NewDataSourceTool, AdvancedVisualizationTool)
2. Update the Tool Selector to consider the new tool
3. Enhance the Task Planner to incorporate the new capability
4. Add relevant prompts and few-shot examples for the LLM

## Contributing
We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

## License
This project is licensed under the  Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
