"""
Example: Basic TTS Synthesis
Demonstrates how to use the TTS system for text-to-speech synthesis
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.synthesizer import TTSSynthesizer


def main():
    # Initialize synthesizer
    # Note: You need to train models first or download pretrained checkpoints
    tts = TTSSynthesizer(
        acoustic_checkpoint='checkpoints/acoustic/acoustic_model_final.pt',
        vocoder_checkpoint='checkpoints/vocoder/generator_final.pt'
    )
    
    # Example 1: Basic synthesis
    print("Example 1: Basic synthesis")
    text = "Hello, this is a test of the neural text to speech system."
    audio = tts.synthesize(text, language='en')
    tts.save_audio(audio, 'outputs/example1_basic.wav')
    
    # Example 2: Adjust pitch (higher pitch)
    print("Example 2: Higher pitch")
    audio = tts.synthesize(text, language='en', pitch_scale=1.2)
    tts.save_audio(audio, 'outputs/example2_high_pitch.wav')
    
    # Example 3: Adjust speed (slower)
    print("Example 3: Slower speech")
    audio = tts.synthesize(text, language='en', speed_scale=0.8)
    tts.save_audio(audio, 'outputs/example3_slow.wav')
    
    # Example 4: Adjust speed (faster)
    print("Example 4: Faster speech")
    audio = tts.synthesize(text, language='en', speed_scale=1.3)
    tts.save_audio(audio, 'outputs/example4_fast.wav')
    
    # Example 5: Adjust energy (louder)
    print("Example 5: Higher energy")
    audio = tts.synthesize(text, language='en', energy_scale=1.5)
    tts.save_audio(audio, 'outputs/example5_loud.wav')
    
    # Example 6: Combined adjustments
    print("Example 6: Combined adjustments")
    audio = tts.synthesize(
        text,
        language='en',
        pitch_scale=0.9,  # Slightly lower pitch
        speed_scale=1.1,  # Slightly faster
        energy_scale=1.2  # Slightly louder
    )
    tts.save_audio(audio, 'outputs/example6_combined.wav')
    
    # Example 7: Long text
    print("Example 7: Long text")
    long_text = """
    The neural text to speech system uses a FastSpeech2 acoustic model
    to generate mel-spectrograms from text, and a HiFi-GAN vocoder
    to convert those spectrograms into high-quality audio.
    The system supports multiple languages and allows fine control
    over pitch, speed, and energy for expressive speech synthesis.
    """
    audio = tts.synthesize(long_text, language='en')
    tts.save_audio(audio, 'outputs/example7_long.wav')
    
    # Example 8: Batch synthesis
    print("Example 8: Batch synthesis")
    texts = [
        "This is the first sentence.",
        "This is the second sentence.",
        "This is the third sentence."
    ]
    audios = tts.synthesize_batch(texts, language='en')
    for i, audio in enumerate(audios):
        tts.save_audio(audio, f'outputs/example8_batch_{i+1}.wav')
    
    print("\nAll examples completed! Check the 'outputs' directory for results.")


if __name__ == '__main__':
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    main()
