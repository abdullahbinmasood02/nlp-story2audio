import grpc
import time
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from api import story_audio_pb2
from api import story_audio_pb2_grpc

# Test stories of different lengths
test_stories = {
    "short": "The dog barked loudly.",
    "medium": "The happy child ran through the park on a sunny day. Birds were singing in the trees.",
    "long": """The little girl got a new puppy for her birthday. She was so excited that she jumped up 
    and down with joy! The puppy wagged its tail and licked her face."""
}

def run_request(story_key, num_concurrent):
    """Run a single request and measure response time"""
    try:
        start_time = time.time()
        
        # Connect to the server
        channel = grpc.insecure_channel("localhost:50051")
        stub = story_audio_pb2_grpc.StoryAudioServiceStub(channel)
        
        # Create a request
        request = story_audio_pb2.StoryRequest(
            story_text=test_stories[story_key],
            voice_type="p225"
        )
        
        # Call the gRPC service
        response = stub.GenerateAudio(request)
        
        end_time = time.time()
        
        # Check if response was successful
        if response.status == "success":
            return end_time - start_time
        else:
            print(f"Error: {response.status}")
            return None
            
    except grpc.RpcError as e:
        print(f"gRPC Error: {e.details()}")
        return None

def test_concurrency(max_concurrent_requests=10):
    """Test the API with increasing concurrent requests"""
    
    # Results dictionary to store response times
    results = {
        "short": [],
        "medium": [],
        "long": []
    }
    
    # Test with increasing concurrent requests
    concurrent_requests = list(range(1, max_concurrent_requests + 1))
    
    for num_concurrent in concurrent_requests:
        print(f"Testing with {num_concurrent} concurrent requests...")
        
        for story_key in test_stories.keys():
            # Create a thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                # Submit requests
                futures = [executor.submit(run_request, story_key, num_concurrent) for _ in range(num_concurrent)]
                
                # Collect results
                response_times = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                # Filter out None values (failed requests)
                response_times = [t for t in response_times if t is not None]
                
                if response_times:
                    # Calculate average response time
                    avg_time = sum(response_times) / len(response_times)
                    results[story_key].append(avg_time)
                else:
                    results[story_key].append(None)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    for story_key, times in results.items():
        # Filter out None values for plotting
        valid_indices = [i for i, t in enumerate(times) if t is not None]
        valid_x = [concurrent_requests[i] for i in valid_indices]
        valid_y = [times[i] for i in valid_indices]
        
        if valid_y:
            plt.plot(valid_x, valid_y, marker='o', label=f"{story_key} story")
    
    plt.xlabel('Number of Concurrent Requests')
    plt.ylabel('Average Response Time (seconds)')
    plt.title('API Performance: Response Time vs. Concurrent Requests')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_test_results.png')
    plt.show()
    
    return results

if __name__ == "__main__":
    test_concurrency(10)