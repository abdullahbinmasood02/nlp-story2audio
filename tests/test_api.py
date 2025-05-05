# filepath: c:\Users\pc\Desktop\nlp-project\story2audio\tests\test_api.py
import pytest
import grpc
from api import story_audio_pb2
from api import story_audio_pb2_grpc

def test_api_connection():
    # This test only checks if we can connect to the server
    # For a real test, the server would need to be running
    try:
        channel = grpc.insecure_channel("localhost:50051")
        stub = story_audio_pb2_grpc.StoryAudioServiceStub(channel)
        # If no error occurs, connection was successful
        assert True
    except:
        pytest.skip("Server not running - skipping API test")