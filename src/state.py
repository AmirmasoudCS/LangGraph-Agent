from langgraph.graph import MessagesState


class AgentState(MessagesState):
    iterations: int        # how many times the critic has asked for a fix
    verdict: str           # "ok", "retry" or "gave_up"
    summary: str           # running summary of the older conversation
    summarized_upto: str   # id of the last message covered by the summary