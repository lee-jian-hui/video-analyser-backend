import inspect
import sys
from typing import List, Any


class ToolDiscovery:
    """Utility class for automatically discovering LangChain tools"""

    @staticmethod
    def discover_tools_in_module(module_name: str) -> List[Any]:
        """
        Automatically discover all @tool decorated functions in a given module

        Args:
            module_name: The module name (e.g., 'agents.vision_agent')

        Returns:
            List of discovered LangChain tools
        """
        if module_name not in sys.modules:
            return []

        current_module = sys.modules[module_name]
        tools = []

        for name, obj in inspect.getmembers(current_module):
            # Check if it's a LangChain tool (has the tool attributes)
            if hasattr(obj, 'name') and hasattr(obj, 'description') and callable(obj):
                # Additional check to ensure it's actually a tool
                if hasattr(obj, 'func') or hasattr(obj, '_name'):
                    tools.append(obj)

        return tools

    @staticmethod
    def discover_tools_in_class(cls) -> List[Any]:
        """
        Discover tools defined in the same module as a class

        Args:
            cls: The class instance or class type

        Returns:
            List of discovered LangChain tools
        """
        module_name = cls.__class__.__module__ if hasattr(cls, '__class__') else cls.__module__
        return ToolDiscovery.discover_tools_in_module(module_name)

    @staticmethod
    def get_tool_names(tools: List[Any]) -> List[str]:
        """Get the names of tools for documentation purposes"""
        return [tool.name for tool in tools if hasattr(tool, 'name')]

    @staticmethod
    def get_tool_descriptions(tools: List[Any]) -> dict:
        """Get tool names and descriptions as a dictionary"""
        return {
            tool.name: tool.description
            for tool in tools
            if hasattr(tool, 'name') and hasattr(tool, 'description')
        }