import pytest
from models.emotion_analyzer import EmotionAnalyzer

def test_emotion_analyzer():
    analyzer = EmotionAnalyzer()
    text = "I am so happy today! But yesterday was a sad day."
    results = analyzer.analyze_text(text)
    assert len(results) == 2
    assert results[0]["emotion"] in ["joy", "happy"]
    assert results[1]["emotion"] in ["sadness", "sad"]
