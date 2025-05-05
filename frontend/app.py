import gradio as gr
import grpc
import sys
import os

# Add the parent directory to path to import the API modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Add api directory to sys.path so the generated modules can find each other
api_dir = os.path.join(parent_dir, "api")
sys.path.append(api_dir)

# Now import the modules
from api import story_audio_pb2
from api import story_audio_pb2_grpc

# Update the generate_audio function to return emotion analysis too
def generate_audio(story_text, voice_type):
    # Connect to the gRPC server
    channel = grpc.insecure_channel("localhost:50051")
    stub = story_audio_pb2_grpc.StoryAudioServiceStub(channel)
    
    # Create a request
    request = story_audio_pb2.StoryRequest(
        story_text=story_text,
        voice_type=voice_type
    )
    
    # Call the gRPC service
    try:
        response = stub.GenerateAudio(request)
        if response.status == "success":
            # Save the audio to a file
            with open("output.wav", "wb") as f:
                f.write(response.audio_content)
            
            # If the response includes emotion analysis
            if hasattr(response, 'emotion_analysis'):
                emotion_data = response.emotion_analysis
                return "output.wav", emotion_data
            else:
                # You could request emotional analysis separately
                # or mock it for display purposes
                segments = story_text.split('. ')
                mock_analysis = [{"text": s, "emotion": "unknown"} for s in segments]
                return "output.wav", mock_analysis
        else:
            return f"Error: {response.status}", None
    except grpc.RpcError as e:
        return f"gRPC Error: {e.details()}", None

# Add complete example stories with different emotional content
example_stories = {
    "Happy Story": "The little girl got a new puppy for her birthday. She was so excited that she jumped up and down with joy! The puppy wagged its tail and licked her face. 'I'll call you Buddy!' she exclaimed happily. Her parents smiled, delighted to see their daughter so thrilled with her new best friend. That night, the puppy curled up at the foot of her bed, and she fell asleep with a huge smile on her face.",
    
    "Sad Story": "The old man sat alone on the bench, looking at the photo of his wife. A tear rolled down his cheek as he remembered their fifty years together. The park where they used to walk every Sunday felt empty without her. 'I miss you every day,' he whispered sadly to the picture. The autumn leaves fell around him, matching his mood as the sun began to set. He carefully placed the photo back in his wallet and slowly made his way home to his quiet house.",
    
    "Scared Story": "The dark shadow moved across the wall. She froze in fear, unable to move as the footsteps grew closer. Her heart pounded in her chest, and she struggled to control her breathing. The floorboards creaked just outside her bedroom door. With trembling hands, she reached for her phone, but it was too far away. Suddenly, the doorknob began to turn slowly. She held her breath, terrified of what might come through the door.",
    
    "Angry Story": "The customer slammed his fist on the counter. 'This is the third time your company has messed up my order!' he shouted furiously. His face turned red as the manager approached. 'Sir, please calm down,' the manager said, which only made things worse. 'Don't tell me to calm down!' he yelled, throwing the receipt in the air. Other customers backed away as his anger filled the room. 'I demand to speak to your supervisor immediately!'",
    
    "Surprised Story": "Maria opened the front door and couldn't believe her eyes. 'SURPRISE!' shouted all her friends and family. Her jaw dropped in astonishment as she took in the decorations and the huge banner reading 'Happy 30th Birthday!' 'But... how did you...?' she stammered, completely shocked. Her husband appeared with a cake, grinning widely. 'You really had no idea, did you?' he asked, delighted by her genuine reaction. Maria put her hands to her cheeks in amazement, still processing the unexpected celebration.",
    
    "Mixed Emotions": """The Lost Teddy Bear

Little Emma loved her teddy bear more than anything in the world. She took it everywhere she went and couldn't sleep without it. It was her best friend.

One sunny day at the park, Emma was playing on the swings when she suddenly realized her teddy bear was missing! "Oh no!" she cried in panic. "Where is my teddy bear?!"

Emma searched frantically all over the playground, looking under the slides and behind the trees. Tears began streaming down her face as she grew more worried.

Her mother noticed her distress and came over. "What's wrong, sweetheart?" she asked gently.

"My teddy bear is gone!" Emma sobbed. "I'll never find him again!"

Together, they retraced Emma's steps through the park. They looked in every bush and behind every bench. Just as the sun was beginning to set, Emma's mother spotted something brown and fuzzy under the picnic table.

"Emma! Look!" she called out excitedly.

Emma ran over and gasped with joy. "My teddy!" she shouted, hugging the bear tightly to her chest. "I'm so happy you're back!"

That night, Emma tucked her teddy bear in extra carefully. "I'll never lose you again," she whispered happily before falling into a peaceful sleep."""
}

def load_example(example_name):
    return example_stories[example_name]

# Update your Gradio interface
with gr.Blocks(title="Story2Audio Generator") as demo:
    gr.Markdown("# Story2Audio Generator")
    gr.Markdown("Enter a story and generate an engaging audio version with emotional inflections.")
    
    with gr.Row():
        with gr.Column():
            example_dropdown = gr.Dropdown(
                choices=list(example_stories.keys()),
                label="Load Example Story"
            )
            story_text = gr.Textbox(
                label="Story Text", 
                lines=10, 
                placeholder="Enter your story here..."
            )
            voice_type = gr.Textbox(
                label="Voice Type (optional)", 
                placeholder="e.g., p225"
            )
            generate_btn = gr.Button("Generate Audio")
        
        with gr.Column():
            audio_output = gr.Audio(label="Generated Audio")
            emotion_analysis = gr.Json(label="Emotion Analysis")
    
    example_dropdown.change(load_example, example_dropdown, story_text)
    generate_btn.click(generate_audio, [story_text, voice_type], [audio_output, emotion_analysis])

if __name__ == "__main__":
    # For older Gradio versions (3.x), use these parameters instead
    demo.launch(enable_queue=True, share=False)