import ast 
import operator
from langchain_core.tools import tool

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression like '12 * (3 + 4)'. Use this for any arithmetic."""
    pass