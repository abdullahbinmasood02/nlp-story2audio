from transformers import pipeline
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    def __init__(self):
        # Use a valid public model for emotion analysis
        logger.info("Initializing emotion analyzer")
        self.model = pipeline(
            "text-classification", 
            model="bhadresh-savani/distilbert-base-uncased-emotion", 
            return_all_scores=True
        )
        logger.info("Emotion analyzer initialized successfully")
        
        # Map model's emotion labels to standardized ones
        self.emotion_map = {
            "joy": "joy",
            "sadness": "sadness", 
            "anger": "anger",
            "fear": "fear",
            "love": "joy",
            "surprise": "surprise"
        }
        
    def analyze_text(self, text):
        """
        Analyze the emotional content of text with more granular detection.
        Returns a list of emotion scores for each sentence.
        """
        logger.info(f"Analyzing text: {text[:50]}...")
        
        # Add exclamation detection for stronger emotions
        def has_exclamation(s):
            return '!' in s
        
        # Add question detection for inquisitive tone
        def has_question(s):
            return '?' in s
            
        # Split text into smaller chunks for more granular emotion detection
        # Using dialog markers and punctuation
        chunks = []
        
        # First split by dialog markers
        dialog_segments = re.split(r'("[^"]*")', text)
        
        for segment in dialog_segments:
            if segment.startswith('"') and segment.endswith('"'):
                # Keep dialog as one chunk with its surrounding quotes
                chunks.append(segment)
            else:
                # Further split non-dialog by sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', segment)
                chunks.extend([s for s in sentences if s.strip()])
        
        results = []
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            logger.info(f"Analyzing chunk: {chunk}")
            # Analyze the chunk
            emotion_scores = self.model(chunk)[0]
            
            # Get the dominant emotion
            dominant_emotion = max(emotion_scores, key=lambda x: x["score"])
            emotion_label = self.emotion_map.get(dominant_emotion["label"], dominant_emotion["label"])
            
            logger.info(f"Dominant emotion: {dominant_emotion['label']} (score: {dominant_emotion['score']})")
            
            # Enhance emotion intensity for exclamations
            intensity_modifier = 1.0
            if has_exclamation(chunk):
                intensity_modifier = 1.5  # Strengthen the emotion for exclamations
            
            # Add additional context for questions
            if has_question(chunk):
                if emotion_label not in ["fear", "surprise"]:
                    emotion_label = "surprise"  # Questions often have a surprised/curious tone
            
            results.append({
                "text": chunk,
                "emotion": emotion_label,
                "score": dominant_emotion["score"] * intensity_modifier,
                "all_scores": emotion_scores
            })
            
        logger.info(f"Analysis complete. Found {len(results)} segments.")
        return results