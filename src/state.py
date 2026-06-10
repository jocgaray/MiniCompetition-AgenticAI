from typing import Any, List, NotRequired, TypedDict

class AgentState(TypedDict):
    raw_data: Any
    schema_ok: bool
    errors: List[str]

    # Use NotRequired for fields that start empty or are filled later
    description: NotRequired[str]
    sentiment_label: NotRequired[str]
    sentiment_score: NotRequired[float]
    reasoning: NotRequired[str]
    features: NotRequired[dict]
    clean_features: NotRequired[dict]