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

def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        return OPERATORS[type[node.op]](left, right)
    if isinstance(node, ast.UnaryOp):
        return OPERATORS[type(node.op)](_evaluate(node.operand))
    return ValueError("Unsupported Expression")

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression such as '12 * (3 + 4)' or '2 ** 10 / 4'.
    Supports + - * / % ** and parentheses. Use this for ANY arithmetic."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_evaluate(tree.body))
    except Exception as e:
        return f"Error: {e}"