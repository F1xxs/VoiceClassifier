## VoiceClassifier

A real-time voice classification tool built in Python.

#### Overview

VoiceClassifier uses audio input from your microphone to extract speaker embeddings (via SpeechBrain) and classify/recognize voices in real time. It is useful for speaker identification and noise-cancelling.  
Key features:

- Live audio capture using `sounddevice`
- Embedding extraction using `torch`, `torchaudio` transforms, and SpeechBrain models
- Cosine-similarity-based classification with a simple interface
- CLI arguments for ease of use

#### Libraries Used

This project leverages the following libraries:

- **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet):** For real-time noise suppression.
- **[SpeechBrain](https://speechbrain.github.io/):** For speaker embedding extraction and other speech processing tasks.
- **[Torchaudio](https://pytorch.org/audio/stable/index.html):** For audio transformations and preprocessing.
- **[Sounddevice](https://python-sounddevice.readthedocs.io/):** For capturing and playing audio in real time.
- **[PyTorch](https://pytorch.org/):** As the core deep learning framework.

#### Installation

**1.** Clone the repository:

```bash
git clone https://github.com/F1xxs/VoiceClassifier.git
cd VoiceClassifier
```

**2.** Install dependencies:

```bash
pip install -r requirements.txt
```

#### Usage

Before running the live voice classification, you first need to **create a speaker embedding** - a numerical "voiceprint" that represents the unique features of a person’s voice.  
This is done using the `embedding.py` script.

Run the script on one or more voice samples:

```bash
python embedding.py audio_files
```

Or run it on a folder:

```bash
python embedding.py ./my_voice_samples/ -o embedding.pt
```

You can provide multiple audio files to improve accuracy - the script will average their embeddings into one voiceprint.  
**Note:** All input files must be:

- **Mono** (1 channel)
- **48kHz** sample rate
- either **.wav** or **.ogg** format

Then run the main script:

```bash
python VoiceClassifier.py [options]
```

**Common options:**

- `--list` : List available audio devices and exit
- `--embedding` : Path to the speaker embedding file
- `--low-threshold` : Low threshold for gate hysteresis
- `--high-threshold` : High threshold for gate hysteresis
- `--chunk-duration` : The duration of each chunk processing in seconds
- `--show-score` : Print the score of each chunk

Example:

```bash
python VoiceClassifier.py --chunk-duration 0.8 --show-score --embedding embedding.pt
```

Once running, the tool will ask you to select the input and output devices. After selection, it listens to the input device, denoises the audio, computes embeddings, compares them against known speaker embeddings, and sends the processed audio to the output device if the score is above the threshold.

#### Prerequisites & Notes

- Python 3.8+ recommended
- Requires a working microphone
- GPU acceleration is optional (CPU is supported)

#### License

This project is licensed under the MIT License ([LICENSE](LICENSE) or http://opensource.org/licenses/MIT).
