from enum import Enum
from pydantic import BaseModel, Field

class SentimentLabel(str, Enum):
    VERY_NEGATIVE = "very_negative"
    SOMEWHAT_NEGATIVE = "somewhat_negative"
    NEUTRAL = "neutral"
    SOMEWHAT_POSITIVE = "somewhat_positive"
    VERY_POSITIVE = "very_positive"


class SentimentAnalysis(BaseModel):
    label: SentimentLabel
    reasoning: str = Field(
        description="Explain why this label was chosen based on the decision table."
    )
