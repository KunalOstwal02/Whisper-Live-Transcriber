# OpenAI Whisper Live Transcriber

Real-time speech-to-text that types directly at your cursor on Ubuntu. Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2-optimized OpenAI Whisper) with GPU acceleration.

## Quick Start

```bash
# 1. Setup (one-time)
chmod +x setup.sh
./setup.sh

# 2. Run
source .venv/bin/activate
python3 transcriber.py

# 3. Press Ctrl+Alt+S to toggle listening on/off
# 4. Speak — text appears at your cursor
# 5. Ctrl+C to quit
```

## How It Works

```
Microphone → Voice Activity Detection → Whisper (GPU) → xdotool type → cursor
             (webrtcvad, ~0ms)          (~200-500ms)     (instant)
```

1. Audio streams continuously from your microphone at 16 kHz
2. **webrtcvad** detects speech onset/offset with 30ms granularity
3. Once you pause (~600ms silence), the speech chunk is sent to **faster-whisper**
4. Transcribed text is typed at your current cursor position via **xdotool**

**Typical end-to-end latency:** ~0.8–1.2s after you finish a phrase.

## CLI Options

```
python3 transcriber.py [OPTIONS]

Options:
  -m, --model          tiny.en | base.en | small.en | medium.en | large-v3
                       (default: small.en)
  -d, --device         cuda | cpu  (default: cuda)
  -c, --compute-type   float16 | int8 | int8_float16 | float32
                       (default: float16)
  --language LANG      Whisper language code, e.g. en, fr, de, zh
                       (default: en). For non-English use large-v3;
                       .en models are English-only.
  --silence-ms N       Silence before transcribing, in ms (default: 600)
  --input-device N     Audio input device index
  --list-devices       List available audio devices and exit
```

### Examples

```bash
# Fastest possible (lower accuracy)
python3 transcriber.py -m tiny.en --silence-ms 400

# Best accuracy (higher latency)
python3 transcriber.py -m medium.en

# CPU-only fallback
python3 transcriber.py -d cpu -c float32 -m base.en

# Use a specific microphone
python3 transcriber.py --list-devices          # find the index
python3 transcriber.py --input-device 3        # use device 3

# Transcribe in French
python3 transcriber.py --language fr -m large-v3

# German, CPU-only
python3 transcriber.py --language de -m large-v3 -d cpu -c float32
```

## Tuning for Your Setup

| GPU          | Recommended model | Expected latency |
|--------------|-------------------|-----------------|
| RTX 3070+    | large-v3 (multilingual) | ~0.8s     |
| RTX 4090     | medium.en         | ~0.3s           |
| RTX 3070/80  | small.en          | ~0.4s           |
| RTX 3060     | small.en          | ~0.6s           |
| GTX 1660     | base.en           | ~0.5s           |
| CPU only     | tiny.en / base.en | ~1-3s           |

**To reduce latency:**
- Use a smaller model (`base.en` or `tiny.en`)
- Lower `--silence-ms` to 400 (may cut off words if you pause mid-sentence)
- Use `int8` compute type: `-c int8`

**To improve accuracy:**
- Use a larger model (`medium.en`)
- Raise `--silence-ms` to 800 (waits longer for complete phrases)

## Troubleshooting

**"xdotool not found"**
```bash
sudo apt install xdotool
```

**No audio input / wrong microphone**
```bash
python3 transcriber.py --list-devices
python3 transcriber.py --input-device <INDEX>
```

**Hotkey not working**
- `pynput` requires X11. On Wayland, you may need to run under XWayland or add yourself to the `input` group:
  ```bash
  sudo usermod -aG input $USER
  # then log out and back in
  ```

**CUDA out of memory**
- Use a smaller model: `-m base.en` or `-m tiny.en`
- Use int8 quantization: `-c int8`

**Text not appearing / garbled output**
- Make sure your target application has keyboard focus
- Some Electron apps (VS Code, Slack) may need a small `xdotool` delay — edit `--delay 0` to `--delay 12` in the script

**Whisper hallucinates (outputs random text during silence)**
- The script filters common hallucinations automatically
- If you see repetitive artefacts, increase `VAD_AGGRESSIVENESS` to 3 in the script

## Wayland Support

On Wayland sessions, the script auto-detects and uses `wtype` instead of `xdotool`:

```bash
sudo apt install wtype
```

## Multilingual

Defaults to English. Pass `--language` with any [Whisper-supported language code](https://github.com/openai/whisper#available-models-and-languages):

```bash
python3 transcriber.py --language fr -m large-v3   # French
python3 transcriber.py --language de -m large-v3   # German
python3 transcriber.py --language zh -m large-v3   # Chinese
```

> **Note:** The `.en` model variants (`tiny.en`, `base.en`, `small.en`, `medium.en`) are English-only distillations and will override `--language` and always produce English output. Use `large-v3` for multilingual transcription.

## Requirements

- Ubuntu 22.04+ (or any Linux with PulseAudio/PipeWire)
- Python 3.10+
- NVIDIA GPU with drivers installed (for CUDA mode)
- ~1 GB disk for the `small.en` model

## License

MIT — see [LICENSE](LICENSE).
