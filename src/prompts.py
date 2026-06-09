# prompts.py
SENTIMENT_ANALYSIS_PROMPT = """
You are a sentiment analyst. Analyze the following property description: "{description}"

Use the following Decision Table:
- "Very Negative": Extreme failure, unlivable, severe damage.
- "Somewhat Negative": Minor issues, disappointment, needs cleanup.
- "Neutral": Factual description, no emotion.
- "Somewhat Positive": Satisfied, good condition, minor improvements needed.
- "Very Positive": Perfect, high-end, luxury, highly recommended.

Output the sentiment label and your reasoning.
"""

FEATURE_EXTRACTION_PROMPT = """
You are a property analysis expert. Extract the following features from the property description below:

1. is_luxury: (boolean) True if the property mentions high-end finishes, premium location, or luxury amenities.
2. needs_renovation: (boolean) True if the description mentions damage, disrepair, or outdated fixtures.
3. amenities: (list[str]) A list of specific amenities mentioned (e.g., "pool", "balcony", "modern kitchen").
4. core_condition: (string) A short summary of the overall physical state of the property.

Property Description: "{description}"

Return the data in a structured JSON format.
"""
