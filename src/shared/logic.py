from .schemas import SentimentLabel

def map_label_to_score(label: SentimentLabel) -> float:
    mapping = {
        SentimentLabel.VERY_NEGATIVE: 0.0,
        SentimentLabel.SOMEWHAT_NEGATIVE: 1.25,
        SentimentLabel.NEUTRAL: 2.5,
        SentimentLabel.SOMEWHAT_POSITIVE: 3.75,
        SentimentLabel.VERY_POSITIVE: 5.0,
    }
    return mapping.get(label, 2.5)
