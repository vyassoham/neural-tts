"""
Command-line interface for Neural TTS
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Neural TTS - Production-grade Text-to-Speech',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic synthesis
  neural-tts "Hello, world!" -o output.wav
  
  # Adjust prosody
  neural-tts "Hello!" -o output.wav --pitch 1.2 --speed 0.9
  
  # Different language
  neural-tts "Hola, mundo!" -o output.wav --language es
  
  # Voice cloning
  neural-tts "Clone this voice" -o output.wav --clone reference.wav
        """
    )
    
    # Required arguments
    parser.add_argument('text', type=str, help='Text to synthesize')
    
    # Output options
    parser.add_argument('-o', '--output', type=str, default='output.wav',
                       help='Output audio file (default: output.wav)')
    
    # Language options
    parser.add_argument('-l', '--language', type=str, default='en',
                       help='Language code (default: en)')
    
    # Prosody control
    parser.add_argument('--pitch', type=float, default=1.0,
                       help='Pitch scale (default: 1.0)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speed scale (default: 1.0)')
    parser.add_argument('--energy', type=float, default=1.0,
                       help='Energy scale (default: 1.0)')
    
    # Voice cloning
    parser.add_argument('--clone', type=str, default=None,
                       help='Reference audio for voice cloning')
    
    # Model options
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model directory')
    
    # Speaker selection
    parser.add_argument('--speaker', type=int, default=0,
                       help='Speaker ID for multi-speaker models (default: 0)')
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup
    try:
        if args.clone:
            from neural_tts import VoiceCloner
            
            print(f"Initializing voice cloner...")
            cloner = VoiceCloner(
                acoustic_checkpoint=f"{args.model_path or 'checkpoints'}/acoustic/acoustic_model_final.pt",
                vocoder_checkpoint=f"{args.model_path or 'checkpoints'}/vocoder/generator_final.pt",
                speaker_encoder_checkpoint=f"{args.model_path or 'checkpoints'}/speaker/encoder_final.pt"
            )
            
            print(f"Cloning voice from: {args.clone}")
            print(f"Synthesizing: {args.text}")
            
            audio = cloner.clone_voice(
                text=args.text,
                reference_audio=args.clone,
                language=args.language,
                pitch_scale=args.pitch,
                speed_scale=args.speed,
                energy_scale=args.energy
            )
            
            cloner.save_audio(audio, args.output)
            
        else:
            from neural_tts import TTS
            
            print(f"Initializing TTS...")
            tts = TTS(model_path=args.model_path)
            
            print(f"Synthesizing: {args.text}")
            
            tts.speak(
                text=args.text,
                language=args.language,
                output_file=args.output,
                speaker_id=args.speaker,
                pitch_scale=args.pitch,
                speed_scale=args.speed,
                energy_scale=args.energy
            )
        
        print(f"✓ Audio saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
