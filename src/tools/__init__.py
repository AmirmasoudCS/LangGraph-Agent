from .calculator import calculator
from .weather import get_weather
from .web_search import web_search
from .datetime import get_current_datetime, time_difference
from .doc_search import search_documents, make_search_tool

# Tools that need nothing but the internet. The benchmark's original categories
# use only these, so their results stay comparable.
BASE_TOOLS = [calculator, get_weather, web_search, get_current_datetime, time_difference]

# Everything the chat agent can use, including document search.
ALL_TOOLS = BASE_TOOLS + [search_documents]