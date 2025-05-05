import unittest
from models.emotion_analyzer import EmotionAnalyzer

class TestEmotionAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = EmotionAnalyzer()
    
    def test_happy_text(self):
        result = self.analyzer.analyze_text("I am so happy today!")
        self.assertEqual(result[0]["emotion"], "joy")
    
    def test_sad_text(self):
        result = self.analyzer.analyze_text("I feel very sad and lonely.")
        self.assertEqual(result[0]["emotion"], "sadness")

if __name__ == "__main__":
    unittest.main()