"""
Speaker embedding enrollment script.
Generates averaged speaker embeddings from multiple audio samples.
The audio samples should be 48khz and mono
"""
import logging
import torch
import torchaudio
import torchaudio.functional as F
from pathlib import Path
from typing import List, Optional
import warnings
import argparse
import sys

warnings.filterwarnings("ignore")

from df.enhance import enhance, init_df 
from speechbrain.inference import EncoderClassifier

# Suppress DeepFilterNet verbose output


class EmbeddingGenerator:
    """Generate speaker embeddings from audio files."""
        
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize embedding generator with models.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        print(f"🔄 Loading models on {device}...")
        
        torch.set_grad_enabled(False)
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_ecapa",
            run_opts={"device": device}
        )
        self.ecapa.eval()
        
        # Load DeepFilterNet for enhancement
        self.df_model, self.df_state, _ = init_df(log_level="NONE")
        self.df_model.to(device)
        
        print("✅ Models loaded successfully")
    
    def get_embedding(self, wav_path: Path) -> Optional[torch.Tensor]:
        """
        Extract speaker embedding from audio file.
        
        Args:
            wav_path: Path to audio file
            
        Returns:
            Normalized speaker embedding tensor or None if processing fails
        """
        try:
            signal, fs = torchaudio.load(str(wav_path))
            signal = signal.to(self.device)
            if fs != 48000:
                raise Exception("the audiofile is not 48kHz!")
            if signal.shape[0] != 1:
                raise Exception("the audiofile is not mono!")
            
            # Enhance audio with DeepFilterNet and encode
            with torch.inference_mode():
                enhanced = enhance(self.df_model, self.df_state, signal)
                enhanced = F.resample(enhanced, 48000, 16000)
                embedding = self.ecapa.encode_batch(enhanced).squeeze(0)
                embedding = torch.nn.functional.normalize(embedding, dim=-1)
            
            print(f"✅ Processed: {wav_path.name}")
            return embedding
            
        except Exception as e:
            print(f"❌ Failed to process {wav_path}: {e}")
            return None
    
    def create_voiceprint(self, audio_files: List[Path], output_path: Path) -> bool:
        """
        Create averaged voiceprint from multiple audio samples.
        
        Args:
            audio_files: List of paths to enrollment audio files
            output_path: Path to save the voiceprint
            
        Returns:
            True if successful, False otherwise
        """
        if not audio_files:
            print("❌ No audio files provided")
            return False
        
        print(f"📁 Processing {len(audio_files)} enrollment files...")
        
        embeddings = []
        for audio_file in audio_files:
            if not audio_file.exists():
                print(f"⚠️ File not found: {audio_file}")
                continue

                
            embedding = self.get_embedding(audio_file)
            if embedding is not None:
                embeddings.append(embedding)
        
        if not embeddings:
            print("❌ No valid embeddings extracted")
            return False

        voiceprint = torch.mean(torch.stack(embeddings), dim=0)
        voiceprint = torch.nn.functional.normalize(voiceprint, dim=-1)
        try:
            torch.save(voiceprint, output_path)
            print(f"✅ Voiceprint saved to {output_path}")
            print(f"📊 Used {len(embeddings)}/{len(audio_files)} files")
            return True
        except Exception as e:
            print(f"❌ Failed to save voiceprint: {e}")
            return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="embedding",
        description="Generate speaker voiceprint from enrollment audio files."
    )
    parser.add_argument(
        "audio_files",
        nargs='+',
        type=Path,
        help="Paths to enrollment audio files (supports wildcards)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("embedding.pt"),
        help="Output path for voiceprint (default: embedding.pt)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    audio_files = []
    for path in args.audio_files:
        if path.is_dir():
            audio_files.extend(path.glob("*.wav"))
            audio_files.extend(path.glob("*.ogg"))
        else:
            audio_files.append(path)
    
    if not audio_files:
        print("❌ No audio files found")
        sys.exit(1)
    generator = EmbeddingGenerator(device=args.device)
    success = generator.create_voiceprint(audio_files, args.output)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()