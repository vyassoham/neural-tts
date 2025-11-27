"""
Example: Voice Cloning
Demonstrates how to clone a voice from reference audio
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.voice_cloner import VoiceCloner


def main():
    # Initialize voice cloner
    # Note: You need trained models and a speaker encoder
    cloner = VoiceCloner(
        acoustic_checkpoint='checkpoints/acoustic/acoustic_model_final.pt',
        vocoder_checkpoint='checkpoints/vocoder/generator_final.pt',
        speaker_encoder_checkpoint='checkpoints/speaker/encoder_final.pt'
    )
    
    # Path to reference audio (5-10 seconds of clean speech)
    reference_audio = 'data/reference_audio.wav'
    
    # Example 1: Clone voice and synthesize
    print("Example 1: Basic voice cloning")
    text = "This sentence is spoken in the cloned voice."
    audio = cloner.clone_voice(text, reference_audio, language='en')
    cloner.save_audio(audio, 'outputs/cloned_voice_1.wav')
    
    # Example 2: Clone with pitch adjustment
    print("Example 2: Cloned voice with higher pitch")
    audio = cloner.clone_voice(
        text,
        reference_audio,
        language='en',
        pitch_scale=1.2
    )
    cloner.save_audio(audio, 'outputs/cloned_voice_2_high_pitch.wav')
    
    # Example 3: Extract and reuse speaker embedding
    print("Example 3: Reusing speaker embedding")
    speaker_embedding = cloner.extract_speaker_embedding(reference_audio)
    
    texts = [
        "First sentence in the cloned voice.",
        "Second sentence in the cloned voice.",
        "Third sentence in the cloned voice."
    ]
    
    for i, text in enumerate(texts):
        audio = cloner.clone_voice_from_embedding(
            text,
            speaker_embedding,
            language='en'
        )
        cloner.save_audio(audio, f'outputs/cloned_voice_3_{i+1}.wav')
    
    # Example 4: Compare speaker similarity
    print("Example 4: Speaker similarity comparison")
    reference_audio_2 = 'data/reference_audio_2.wav'
    
    # Compare same speaker
    similarity_same = cloner.compare_speakers(reference_audio, reference_audio)
    print(f"Similarity (same speaker): {similarity_same:.4f}")
    
    # Compare different speakers
    if os.path.exists(reference_audio_2):
        similarity_diff = cloner.compare_speakers(reference_audio, reference_audio_2)
        print(f"Similarity (different speakers): {similarity_diff:.4f}")
    
    print("\nVoice cloning examples completed!")


if __name__ == '__main__':
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    main()
