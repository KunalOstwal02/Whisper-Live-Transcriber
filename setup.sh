#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Whisper Live Transcriber — one-click setup
#  Tested on Ubuntu 22.04 / 24.04 with NVIDIA GPU
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -euo pipefail

VENV_DIR=".venv"

echo "═══════════════════════════════════════════════════"
echo "  Whisper Live Transcriber — Setup"
echo "═══════════════════════════════════════════════════"
echo ""

# ── 1. System packages ────────────────────────────────
echo "▸ Installing system dependencies…"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    xdotool \
    portaudio19-dev \
    python3-dev \
    python3-venv \
    build-essential \
    ffmpeg \
    2>/dev/null
echo "  ✔ System packages installed"

# ── 2. Virtual environment ────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "▸ Creating virtual environment…"
    python3 -m venv "$VENV_DIR"
    echo "  ✔ Virtual environment created at $VENV_DIR/"
else
    echo "  ✔ Virtual environment already exists"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ── 3. Python packages ───────────────────────────────
echo "▸ Installing Python packages…"
pip install --upgrade pip -q

pip install -q -r requirements.txt

echo "  ✔ Python packages installed"

# ── 4. CUDA check ─────────────────────────────────────
echo ""
echo "▸ Checking GPU…"
if python3 -c "import ctranslate2; print('  CTranslate2:', ctranslate2.__version__)" 2>/dev/null; then
    if nvidia-smi &>/dev/null; then
        echo "  ✔ NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    else
        echo "  ⚠ nvidia-smi not found — will fall back to CPU"
        echo "    Install NVIDIA drivers for GPU acceleration."
    fi
else
    echo "  ⚠ CTranslate2 import failed — check installation"
fi

# ── 5. Microphone check ──────────────────────────────
echo ""
echo "▸ Checking audio devices…"
python3 -c "
import sounddevice as sd
default = sd.query_devices(kind='input')
print(f\"  Default input: {default['name']}  ({int(default['default_samplerate'])} Hz, {default['max_input_channels']} ch)\")
" 2>/dev/null || echo "  ⚠ Could not query audio devices"

# ── 6. Pre-download model ────────────────────────────
echo ""
echo "▸ Pre-downloading Whisper model (small.en)…"
echo "  (This runs once; ~500 MB download)"
python3 -c "
from faster_whisper import WhisperModel
model = WhisperModel('small.en', device='cpu', compute_type='float32')
print('  ✔ Model cached')
" 2>/dev/null || echo "  ⚠ Model download skipped — will download on first run"

# ── Done ──────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅  Setup complete!"
echo ""
echo "  To run:"
echo "    source $VENV_DIR/bin/activate"
echo "    python3 transcriber.py"
echo ""
echo "  Then press Ctrl+Alt+S to start dictating."
echo "═══════════════════════════════════════════════════"
