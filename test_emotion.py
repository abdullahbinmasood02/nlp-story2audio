from models.emotion_analyzer import EmotionAnalyzer

def test_emotion_analyzer():
    analyzer = EmotionAnalyzer()
    
    test_text = "I am so happy today! I got a new puppy."
    results = analyzer.analyze_text(test_text)
    
    print("=== Test Results ===")
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Emotion: {result['emotion']}")
        print(f"Score: {result['score']}")
        print("---")

if __name__ == "__main__":
    test_emotion_analyzer()