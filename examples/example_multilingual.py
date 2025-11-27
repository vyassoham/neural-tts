"""
Example: Multilingual TTS
Demonstrates synthesis in multiple languages
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.synthesizer import TTSSynthesizer


def main():
    # Initialize synthesizer
    tts = TTSSynthesizer(
        acoustic_checkpoint='checkpoints/acoustic/acoustic_model_final.pt',
        vocoder_checkpoint='checkpoints/vocoder/generator_final.pt'
    )
    
    # Multilingual examples
    examples = [
        ('en', "Hello, how are you today?"),
        ('es', "Hola, ¿cómo estás hoy?"),
        ('fr', "Bonjour, comment allez-vous aujourd'hui?"),
        ('de', "Hallo, wie geht es dir heute?"),
        ('zh', "你好，你今天怎么样？"),
        ('ja', "こんにちは、今日はどうですか？"),
        ('ko', "안녕하세요, 오늘 어떻게 지내세요?"),
        ('ru', "Здравствуйте, как дела сегодня?"),
        ('ar', "مرحبا، كيف حالك اليوم؟"),
        ('hi', "नमस्ते, आज आप कैसे हैं?"),
    ]
    
    print("Generating multilingual speech samples...")
    
    for lang_code, text in examples:
        print(f"Synthesizing {lang_code}: {text}")
        
        try:
            audio = tts.synthesize(text, language=lang_code)
            output_path = f'outputs/multilingual_{lang_code}.wav'
            tts.save_audio(audio, output_path)
            print(f"  ✓ Saved to {output_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\nMultilingual synthesis completed!")


if __name__ == '__main__':
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    main()
