import torch
import sounddevice as sd
import numpy as np
import torchaudio
import torchaudio.transforms as T
from speechbrain.inference import EncoderClassifier
from torch.nn import CosineSimilarity
import argparse
import signal
import sys
import warnings
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

warnings.filterwarnings("ignore")

from df.enhance import enhance, init_df 

logging.getLogger("DF").setLevel(logging.ERROR)

# -------------------------------
# Configuration
# -------------------------------
@dataclass
class Config:
    """Audio processing configuration."""
    sample_rate: int = 48000
    chunk_duration: float = 0.8
    low_threshold: float = 0.25
    high_threshold: float = 0.30
    show_score : bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def blocksize(self) -> int:
        return int(self.sample_rate * self.chunk_duration)
    
    @property
    def resample_size(self) -> int:
        return int(self.blocksize * 16000 // self.sample_rate)


# -------------------------------
# Audio Processor Class
# -------------------------------
class VoiceGate:
    """Real-time voice filtering using speaker verification."""
    
    def __init__(self, config: Config):
        self.config = config
        self.gate_on = False
        
        # Initialize models
        torch.set_grad_enabled(False)
        self._init_models()
        self._init_buffers()
        
    def _init_models(self):
        """Initialize ECAPA-TDNN and DeepFilterNet models."""
        print("🔄 Loading models...")
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_ecapa",
            run_opts={"device": self.config.device}
        )
        self.ecapa.eval()
        
        self.df_model, self.df_state, _ = init_df(log_level="NONE")
        self.df_model.to(self.config.device)
        
        self.cosine_sim = CosineSimilarity(dim=-1)
        self.resampler = T.Resample(
            orig_freq=self.config.sample_rate, 
            new_freq=16000
        ).to(self.config.device)
        
        self.target_embedding: Optional[torch.Tensor] = None
        print("✅ Models loaded")
    
    def _init_buffers(self):
        """Preallocate GPU and pinned CPU buffers."""
        cfg = self.config
        self.sig_48 = torch.empty(1, cfg.blocksize, device=cfg.device)
        self.sig_16 = torch.empty(1, cfg.resample_size, device=cfg.device)
        self.cpu_in = torch.empty(cfg.blocksize, dtype=torch.float32, pin_memory=True)
        self.cpu_out = torch.empty(cfg.blocksize, dtype=torch.float32, pin_memory=True)
        self.zeros_48 = torch.zeros(cfg.blocksize, dtype=torch.float32, device=cfg.device)
    
    def load_embedding(self, path: Path) -> bool:
        """Load target speaker embedding from file."""
        try:
            emb = torch.load(path, map_location=self.config.device, weights_only=True)
            self.target_embedding = torch.nn.functional.normalize(emb.to(self.config.device), dim=-1)
            print(f"✅ Loaded embedding from {path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load embedding: {e}")
            return False
    
    def process_chunk(self, indata: np.ndarray, outdata: np.ndarray, 
                      frames: int, time_info, status):
        """Audio callback for real-time processing."""
        if status:
            print(f"⚠️ Stream status: {status}")
        
        # Copy input to GPU
        np.copyto(self.cpu_in.numpy(), indata[:, 0])
        self.sig_48.copy_(self.cpu_in.unsqueeze(0), non_blocking=False)
        
        with torch.inference_mode():
            # Enhance audio with DeepFilterNet
            enhanced = enhance(self.df_model, self.df_state, self.sig_48).to(Config.device)
            
            # Resample for ECAPA-TDNN (expects 16kHz)
            sig_16_tmp = self.resampler(enhanced)
            self.sig_16.copy_(sig_16_tmp)
            
            # Compute speaker embedding and similarity score
            emb = self.ecapa.encode_batch(self.sig_16).squeeze(0)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            score = self.cosine_sim(emb, self.target_embedding).item()
            
            # Smooth score and apply hysteresis
            
            self._update_gate(score)
            
            # Select output (enhanced audio or silence)
            out_src = enhanced.squeeze(0) if self.gate_on else self.zeros_48
            self.cpu_out.copy_(out_src, non_blocking=False)
        
            # Print status periodically
            if self.config.show_score:
                print(f"🎯 score={score:.3f} gate={'ON' if self.gate_on else 'OFF'}")
        
        outdata[:, 0] = self.cpu_out.numpy()
    
    def _update_gate(self, score: float):
        """Update gate state with hysteresis."""
        if self.gate_on:
            if score <= self.config.low_threshold:
                self.gate_on = False
        else:
            if score >= self.config.high_threshold:
                self.gate_on = True


# -------------------------------
# Device Selection Utilities
# -------------------------------
class DeviceSelector:
    """Helper class for audio device selection."""
    
    SKIP_KEYWORDS = {"pipewire", "settings", "dispatcher", "hdmi", "nvidia", "dummy"}
    
    def __init__(self):
        self.device_map = {}
    
    @staticmethod
    def _is_usable(name: str) -> bool:
        """Check if device name should be shown to user."""
        return not any(k in name.lower() for k in DeviceSelector.SKIP_KEYWORDS)
    
    def list_devices(self):
        """Display filtered list of input and output devices."""
        devices = sd.query_devices()
        index = 1
        
        print("\n📥 Available input devices:")
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and self._is_usable(dev["name"]):
                print(f"  [{index}] {dev['name']}")
                self.device_map[index] = idx
                index += 1
        
        print("\n📤 Available output devices:")
        for idx, dev in enumerate(devices):
            if dev["max_output_channels"] > 0 and self._is_usable(dev["name"]):
                print(f"  [{index}] {dev['name']}")
                self.device_map[index] = idx
                index += 1
        print()
    
    def choose_device(self, prompt: str, mode: str) -> int:
        """Prompt user to select a device by index."""
        while True:
            try:
                choice = int(input(prompt))
                if choice not in self.device_map:
                    raise ValueError("Invalid index")
                
                dev_idx = self.device_map[choice]
                dev = sd.query_devices(dev_idx)
                
                if mode == 'input' and dev['max_input_channels'] == 0:
                    print("❌ That device has no input channels. Try again.")
                    continue
                if mode == 'output' and dev['max_output_channels'] == 0:
                    print("❌ That device has no output channels. Try again.")
                    continue
                
                return dev_idx
            except (ValueError, KeyError):
                print("⚠️ Invalid selection, please enter a valid index.")


# -------------------------------
# Main Application
# -------------------------------
def signal_handler(signum, frame):
    """Clean up on SIGINT."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n👋 Shutting down...")
    sys.exit(0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="VoiceClassifier",
        description="Real-time voice filtering using speaker verification."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "--embedding",
        type=Path,
        default=Path("embedding.pt"),
        help="Path to speaker embedding file (default: embedding.pt)"
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.25,
        help="Low threshold for gate hysteresis (default: 0.25)"
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.30,
        help="High threshold for gate hysteresis (default: 0.30)"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.8,
        help="The duration of each chunk processing in seconds (default: 0.8)"
    )
    parser.add_argument(
        "--show-score",
        action="store_true",
        help="Will print the score each chunk"
    )
    return parser.parse_args()


def main():
    """Main application entry point."""
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
    
    # Handle --list flag
    if args.list:
        selector = DeviceSelector()
        selector.list_devices()
        return
    
    # Initialize configuration and voice gate
    config = Config(
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        chunk_duration=args.chunk_duration,
        show_score=args.show_score

    )
    voice_gate = VoiceGate(config)
    
    # Load speaker embedding
    if not voice_gate.load_embedding(args.embedding):
        print(f"❌ Could not load embedding from {args.embedding}")
        return
    
    # Device selection
    selector = DeviceSelector()
    selector.list_devices()
    
    input_dev = selector.choose_device("🎙️ Enter input device index: ", "input")
    output_dev = selector.choose_device("🔈 Enter output device index: ", "output")
    
    print(f"\n✅ Input:  {sd.query_devices(input_dev)['name']}")
    print(f"✅ Output: {sd.query_devices(output_dev)['name']}\n")
    
    # Start streaming
    with sd.Stream(
        device=(input_dev, output_dev),
        samplerate=config.sample_rate,
        channels=1,
        dtype='float32',
        blocksize=config.blocksize,
        callback=voice_gate.process_chunk
    ):
        torch.cuda.empty_cache()
        input("🎙️ Streaming... Press Enter or Ctrl + C to stop\n")


if __name__ == "__main__":
    main()