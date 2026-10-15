from datetime import datetime
from langchain_core.tools import tool

@tool
def get_current_datetime() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%A, %Y-%m-%d %H:%M")