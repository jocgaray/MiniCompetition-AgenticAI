from logic import map_label_to_score
from schemas import SentimentAnalysis

# Inside your loop:
raw_result = await asyncio.wait_for(finished_task, timeout=120.0)
# 'raw_result' is now your validated SentimentAnalysis object
score = map_label_to_score(raw_result.label)
