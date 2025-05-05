# filepath: c:\Users\pc\Desktop\nlp-project\story2audio\api\server.py
import grpc
import time
import sys
import os
from concurrent import futures

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now import the modules
import api.story_audio_pb2 as story_audio_pb2
import api.story_audio_pb2_grpc as story_audio_pb2_grpc
from models.emotion_analyzer import EmotionAnalyzer
from models.tts_generator import TTSGenerator

class StoryAudioServicer(story_audio_pb2_grpc.StoryAudioServiceServicer):
    def __init__(self):
        self.emotion_analyzer = EmotionAnalyzer()
        self.tts_generator = TTSGenerator()
    
    def GenerateAudio(self, request, context):
        try:
            # Analyze emotions in the story
            analyzed_segments = self.emotion_analyzer.analyze_text(request.story_text)
            
            # Generate audio with emotional inflections
            audio_data = self.tts_generator.generate_audio(
                analyzed_segments, 
                request.voice_type
            )
            
            # Create response with both audio and emotion analysis
            response = story_audio_pb2.AudioResponse(
                status="success",
                audio_content=audio_data,
                format="wav",
                sample_rate=22050
            )
            
            # Add emotion analysis data to response
            for segment in analyzed_segments:
                emotion_data = response.emotion_analysis.add()
                emotion_data.text = segment["text"]
                emotion_data.emotion = segment["emotion"]
                emotion_data.score = float(segment["score"])
            
            return response
            
        except Exception as e:
            return story_audio_pb2.AudioResponse(
                status=f"error: {str(e)}",
                audio_content=b"",
                format="wav",
                sample_rate=22050
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    story_audio_pb2_grpc.add_StoryAudioServiceServicer_to_server(
        StoryAudioServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started on port 50051")
    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__": 
    serve()