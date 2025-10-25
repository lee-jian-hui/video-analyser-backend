from langchain.tools import tool




# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b



def inject_llm_tools(model, additional_tools=None):
    # Augment the LLM with tools
    base_tools = [add, multiply, divide]

    # Add any additional tools from agents
    if additional_tools:
        base_tools.extend(additional_tools)

    tools_by_name = {tool.name: tool for tool in base_tools}
    model_with_tools = model.bind_tools(base_tools)
    return model_with_tools, tools_by_name



