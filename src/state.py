from typing import Any, List, NotRequired, TypedDict

class AgentState(TypedDict):
    raw_data: Any
    schema_ok: bool
    errors: List[str]
    predictions: NotRequired[list[int]]

    # Populated by the agent during processing
    messages: NotRequired[list]
    clean_features: NotRequired[Any]
    has_region_price: NotRequired[bool]
    has_sentiment: NotRequired[bool]