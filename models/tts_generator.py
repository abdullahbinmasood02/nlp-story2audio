import io
import numpy as np
import soundfile as sf
from TTS.api import TTS
import os
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSGenerator:
    def __init__(self):
        # Initialize TTS model
        logger.info("Initializing TTS Generator")
        self.tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
        self.default_speaker = "p225"  # Default VCTK speaker
        self.sample_rate = 22050  # Standard sample rate for this model
        logger.info("TTS Generator initialized successfully")
    
    def adjust_speech_params(self, emotion):
        """
        Map emotions to speech parameters with much more dramatic adjustments
        """
        # Default parameters (neutral)
        params = {
            "speed": 1.0,
        }
        
        # Make much more dramatic adjustments for each emotion
        if emotion in ["joy", "happiness"]:
            params["speed"] = 1.15  # Faster for happy
        elif emotion == "sadness":
            params["speed"] = 0.85  # Much slower for sad
        elif emotion == "anger":
            params["speed"] = 1.2  # Fast for angry
        elif emotion == "fear":
            params["speed"] = 1.1  # Slightly faster for fear
        elif emotion == "surprise":
            params["speed"] = 1.1  # Slightly faster for surprise
            
        logger.info(f"Adjusted parameters for {emotion}: {params}")
        return params
    
    def apply_post_processing(self, audio_data, emotion):
        """
        Apply more dramatic post-processing effects based on emotion
        """
        try:
            import librosa
            import pyrubberband as pyrb
            
            # Make a copy to avoid modifying the original
            processed = np.copy(audio_data)
            
            # Apply different effects based on emotion
            if emotion in ["joy", "happiness"]:
                # Increase pitch for happiness
                processed = pyrb.pitch_shift(processed, self.sample_rate, 2.0)
                # Add a slight vibrato for joy
                processed = self._add_vibrato(processed, 5.0, 0.15)
                
            elif emotion == "sadness":
                # Lower pitch for sadness
                processed = pyrb.pitch_shift(processed, self.sample_rate, -2.0)
                # Add slight tremolo for sadness
                processed = self._add_tremolo(processed, 3.0, 0.2)
                
            elif emotion == "anger":
                # Add slight distortion for anger
                processed = self._add_distortion(processed, 0.3)
                # Increase volume for anger
                processed = processed * 1.3
                # Clip to prevent distortion
                processed = np.clip(processed, -1.0, 1.0)
                
            elif emotion == "fear":
                # Add tremolo for fear
                processed = self._add_tremolo(processed, 6.0, 0.25)
                # Add slight echo for fear
                processed = self._add_echo(processed, 0.3, 0.4)
                
            elif emotion == "surprise":
                # Higher pitch for surprise
                processed = pyrb.pitch_shift(processed, self.sample_rate, 3.0)
                
            logger.info(f"Applied post-processing for {emotion}")
            return processed
        except ImportError as e:
            logger.warning(f"Post-processing skipped due to missing dependency: {e}")
            return audio_data
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return audio_data
    
    def _add_vibrato(self, audio, rate=5.0, depth=0.1):
        """Add vibrato effect (pitch oscillation)"""
        length = len(audio)
        lfo = depth * np.sin(2 * np.pi * rate * np.arange(length) / self.sample_rate)
        indices = np.arange(length) + lfo * self.sample_rate / 20
        indices = np.clip(indices, 0, length - 1).astype(np.int32)
        return audio[indices]
    
    def _add_tremolo(self, audio, rate=5.0, depth=0.2):
        """Add tremolo effect (amplitude oscillation)"""
        length = len(audio)
        lfo = 1.0 + depth * np.sin(2 * np.pi * rate * np.arange(length) / self.sample_rate)
        return audio * lfo
    
    def _add_distortion(self, audio, amount=0.2):
        """Add distortion effect"""
        return np.tanh(audio * (1 + amount * 10)) / (1 + amount)
    
    def _add_echo(self, audio, delay=0.2, decay=0.5):
        """Add echo effect"""
        delay_samples = int(delay * self.sample_rate)
        output = np.copy(audio)
        output[delay_samples:] += decay * audio[:-delay_samples]
        return output
    
    def generate_audio(self, analyzed_segments, voice_type=None):
        """
        Generate audio from analyzed text segments with emotional inflection
        """
        logger.info(f"Generating audio for {len(analyzed_segments)} segments")
        
        if not voice_type:
            voice_type = self.default_speaker
        
        # Use temporary files for better audio quality
        temp_files = []
        
        try:
            # Process each segment separately
            for i, segment in enumerate(analyzed_segments):
                if not segment["text"].strip():
                    continue
                    
                emotion = segment["emotion"]
                text = segment["text"]
                
                logger.info(f"Processing segment {i+1}/{len(analyzed_segments)}: '{text[:30]}...' with emotion: {emotion}")
                
                # Get adjusted parameters
                params = self.adjust_speech_params(emotion)
                
                # Create a temporary file for this segment
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_files.append(temp_file.name)
                temp_file.close()
                
                # Generate speech with adjusted parameters
                self.tts.tts_to_file(
                    text=text,
                    file_path=temp_file.name,
                    speaker=voice_type,
                    speed=params["speed"]
                )
                
                # Load the generated audio for post-processing
                audio_data, _ = sf.read(temp_file.name)
                
                # Apply emotional post-processing
                processed_audio = self.apply_post_processing(audio_data, emotion)
                
                # Write the processed audio back
                sf.write(temp_file.name, processed_audio, self.sample_rate)
            
            # Concatenate all segments
            combined_audio = []
            for temp_file in temp_files:
                audio_data, _ = sf.read(temp_file)
                combined_audio.append(audio_data)
                # Add a small silence between segments
                silence = np.zeros(int(self.sample_rate * 0.3))
                combined_audio.append(silence)
                
            # Convert concatenated audio to bytes
            if combined_audio:
                final_audio = np.concatenate(combined_audio)
                buffer = io.BytesIO()
                sf.write(buffer, final_audio, self.sample_rate, format="WAV")
                buffer.seek(0)
                audio_content = buffer.read()
                
                logger.info("Audio generation complete")
                return audio_content
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return None
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Error removing temp file {temp_file}: {e}")