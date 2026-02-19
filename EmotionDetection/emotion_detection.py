import requests 
import json

# Endpoint and headers provided in the assignment
_EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
_EMOTION_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}

def emotion_detector(text_to_analyze: str) -> str:
  
    payload = {"raw_document": {"text": text_to_analyze}}

    # Send POST request
    response = requests.post(
        _EMOTION_URL,
        headers=_EMOTION_HEADERS,
        json=payload,
        timeout=30
    )

    
    resp_dict = json.loads(response.text)

    emotions_path = resp_dict.get("document", {}).get("emotion", {})
    anger   = float(emotions_path.get("anger",   0.0))
    disgust = float(emotions_path.get("disgust", 0.0))
    fear    = float(emotions_path.get("fear",    0.0))
    joy     = float(emotions_path.get("joy",     0.0))
    sadness = float(emotions_path.get("sadness", 0.0))

    scores = {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness
    }
    dominant_emotion = max(scores, key=scores.get)

 
    output = {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": dominant_emotion
    }
    return output

