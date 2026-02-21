"""
server.py

Module to interact with Watson Emotion API.
Provides a function to analyze text and return emotion scores.
"""

from typing import Dict, Optional
import requests

# Constants for API
EMOTION_URL: str = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
EMOTION_HEADERS: Dict[str, str] = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def emotion_detector(text_to_analyze: str) -> Dict[str, Optional[float]]:
    """
    Sends text to Watson Emotion API and returns a dictionary with emotion scores.
    Handles blank inputs and HTTP 400 responses by returning None for all emotions.

    Args:
        text_to_analyze (str): Text to analyze.

    Returns:
        Dict[str, Optional[float]]: Dictionary containing emotion scores and dominant emotion.
    """
    empty_scores: Dict[str, Optional[float]] = {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None
    }

    # Early return for blank input
    if not text_to_analyze or text_to_analyze.strip() == "":
        return empty_scores

    payload: Dict[str, Dict[str, str]] = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(
            EMOTION_URL,
            headers=EMOTION_HEADERS,
            json=payload,
            timeout=30
        )

        if response.status_code == 400:
            return empty_scores

        response.raise_for_status()
        resp_dict: Dict = response.json()
        emotions: Dict[str, float] = resp_dict.get("document", {}).get("emotion", {})

        scores: Dict[str, Optional[float]] = {
            "anger": emotions.get("anger"),
            "disgust": emotions.get("disgust"),
            "fear": emotions.get("fear"),
            "joy": emotions.get("joy"),
            "sadness": emotions.get("sadness")
        }

        dominant_emotion: Optional[str] = None
        if any(value is not None for value in scores.values()):
            dominant_emotion = max(
                scores, key=lambda k: scores[k] if scores[k] is not None else -1
            )

        scores["dominant_emotion"] = dominant_emotion

        return scores

    except requests.exceptions.RequestException:
        return empty_scores