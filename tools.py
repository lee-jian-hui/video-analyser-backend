from langchain.tools import tool



def inject_llm_tools(model, additional_tools=None):
    # Augment the LLM with tools
    base_tools = []

    # Add any additional tools from agents
    if additional_tools:
        base_tools.extend(additional_tools)

    tools_by_name = {tool.name: tool for tool in base_tools}
    
    model_with_tools = model.bind_tools(base_tools)
    return model_with_tools, tools_by_name



