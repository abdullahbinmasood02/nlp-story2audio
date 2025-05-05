import unittest
import numpy as np
from models.tts_generator import TTSGenerator
import tempfile
import os
import soundfile as sf

class TestTTSGenerator(unittest.TestCase):
    def setUp(self):
        self.tts = TTSGenerator()
        
    def test_adjust_speech_params(self):
        # Test different emotions
        params = self.tts.adjust_speech_params("joy")
        self.assertGreater(params["speed"], 1.0)  # Joy should be faster
        
        params = self.tts.adjust_speech_params("sadness")
        self.assertLess(params["speed"], 1.0)  # Sadness should be slower
        
        params = self.tts.adjust_speech_params("anger")
        self.assertGreater(params["speed"], 1.0)  # Anger should be faster
        
    def test_audio_effects(self):
        # Create a simple sine wave as test audio
        sr = 22050
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sr * duration), False)
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Test vibrato
        with_vibrato = self.tts._add_vibrato(test_audio, rate=5.0, depth=0.1)
        self.assertEqual(len(with_vibrato), len(test_audio))
        self.assertFalse(np.array_equal(with_vibrato, test_audio))  # Should be modified
        
        # Test tremolo
        with_tremolo = self.tts._add_tremolo(test_audio, rate=5.0, depth=0.2)
        self.assertEqual(len(with_tremolo), len(test_audio))
        self.assertFalse(np.array_equal(with_tremolo, test_audio))  # Should be modified
        
    def test_generate_audio(self):
        # Test with a simple segment
        test_segments = [
            {"text": "This is a test.", "emotion": "neutral", "score": 0.9}
        ]
        
        # Generate audio
        audio_content = self.tts.generate_audio(test_segments)
        self.assertIsNotNone(audio_content)
        
        # Save to temporary file for validation
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        
        try:
            with open(temp_file.name, 'wb') as f:
                f.write(audio_content)
                
            # Check if file exists and is valid audio
            self.assertTrue(os.path.exists(temp_file.name))
            audio_data, sr = sf.read(temp_file.name)
            self.assertGreater(len(audio_data), 0)
            self.assertEqual(sr, 22050)
            
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

if __name__ == "__main__":
    unittest.main()