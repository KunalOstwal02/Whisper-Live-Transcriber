#!/usr/bin/env python3
"""
Whisper Live Transcriber — real-time speech → text at your cursor.

Toggle ON/OFF : Ctrl + Alt + S
Quit           : Ctrl + C

Architecture:
  Audio capture (sounddevice) → VAD gate (webrtcvad) → Transcribe (faster-whisper) → Type (xdotool/wtype)
"""

import argparse
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from pynput import keyboard

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Defaults (override via CLI flags)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFAULT_MODEL = "small.en"       # tiny.en | base.en | small.en | medium.en
DEFAULT_DEVICE = "cuda"          # cuda | cpu
DEFAULT_COMPUTE = "float16"      # float16 | int8 | int8_float16 | float32
DEFAULT_LANGUAGE = "en"
VAD_AGGRESSIVENESS = 3           # 0 (least) → 3 (most aggressive noise rejection)
SAMPLE_RATE = 16_000             # Whisper expects 16 kHz
FRAME_MS = 30                    # webrtcvad frame size: 10 | 20 | 30 ms
SILENCE_MS = 600                 # silence before we flush speech to Whisper
MIN_SPEECH_MS = 250              # ignore utterances shorter than this
MAX_SPEECH_S = 30                # force-flush if someone talks nonstop

# Derived constants
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
SILENCE_FRAMES = int(SILENCE_MS / FRAME_MS)
MIN_SPEECH_FRAMES = int(MIN_SPEECH_MS / FRAME_MS)
MAX_SPEECH_FRAMES = int(MAX_SPEECH_S * 1000 / FRAME_MS)

# Known Whisper hallucination artefacts
HALLUCINATIONS = {
    "", ".", "you", "thank you.", "thanks for watching.",
    "thanks for watching!", "thank you for watching.",
    "thank you for watching!", "the end.", "the end",
    "subscribe", "like and subscribe", "(music)",
    "[music]", "[blank_audio]", "[silence]",
    "...", "bye.", "bye!", "goodbye.",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Colour helpers for terminal feedback
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def cprint(msg: str, end="\n"):
    sys.stdout.write(msg + end)
    sys.stdout.flush()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Core transcriber
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class LiveTranscriber:

    def __init__(self, model_size: str, device: str, compute_type: str, language: str = DEFAULT_LANGUAGE):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language

        self.is_active = False
        self.audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self.model: WhisperModel | None = None
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._pressed: set = set()
        self._lock = threading.Lock()

        # Detect X11 vs Wayland
        session = os.environ.get("XDG_SESSION_TYPE", "x11").lower()
        self._wayland = "wayland" in session

    # ── model loading ──────────────────────────────────
    def load_model(self):
        cprint(f"{DIM}⏳  Loading '{self.model_size}' on {self.device} ({self.compute_type})…{RESET}")
        t0 = time.time()
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        cprint(f"{GREEN}✔  Model ready in {time.time() - t0:.1f}s{RESET}")

    # ── typing at cursor ───────────────────────────────
    def _type_text(self, text: str):
        text = text.strip()
        if not text:
            return
        out = text + " "
        try:
            if self._wayland:
                subprocess.run(["wtype", "--", out], check=False, timeout=5)
            else:
                subprocess.run(
                    ["xdotool", "type", "--delay", "0", "--clearmodifiers", "--", out],
                    check=False,
                    timeout=5,
                )
        except FileNotFoundError:
            cprint(f"\n{RED}✘  {'wtype' if self._wayland else 'xdotool'} not found — install it.{RESET}")
        except subprocess.TimeoutExpired:
            pass

    # ── transcription ──────────────────────────────────
    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            beam_size=1,
            language=self.language,
            vad_filter=False,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text for seg in segments).strip()
        # Filter hallucinations
        if text.lower() in HALLUCINATIONS:
            return ""
        return text

    # ── hotkey toggle ──────────────────────────────────
    def toggle(self):
        with self._lock:
            self.is_active = not self.is_active
        if self.is_active:
            cprint(f"\r{GREEN}{BOLD}● LISTENING{RESET}    {DIM}(Ctrl+Alt+S to stop){RESET}   ", end="")
        else:
            # Drain leftover audio
            while not self.audio_q.empty():
                try:
                    self.audio_q.get_nowait()
                except queue.Empty:
                    break
            cprint(f"\r{RED}○ PAUSED{RESET}       {DIM}(Ctrl+Alt+S to start){RESET}   ", end="")

    def _on_press(self, key):
        self._pressed.add(key)
        if (
            {keyboard.Key.ctrl_l, keyboard.Key.alt_l}.issubset(self._pressed)
            and key == keyboard.KeyCode.from_char("s")
        ) or (
            {keyboard.Key.ctrl_r, keyboard.Key.alt_r}.issubset(self._pressed)
            and key == keyboard.KeyCode.from_char("s")
        ) or (
            # Also support mixed left/right combos
            any(k in self._pressed for k in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r))
            and any(k in self._pressed for k in (keyboard.Key.alt_l, keyboard.Key.alt_r))
            and key == keyboard.KeyCode.from_char("s")
        ):
            self.toggle()

    def _on_release(self, key):
        self._pressed.discard(key)

    # ── audio callback ─────────────────────────────────
    def _audio_cb(self, indata, frames, time_info, status):
        if self.is_active:
            self.audio_q.put(indata[:, 0].copy())

    # ── main processing loop ───────────────────────────
    def _process_loop(self):
        speech_buf: list[np.ndarray] = []
        triggered = False
        silence_n = 0
        speech_n = 0
        ring = np.array([], dtype=np.float32)

        while True:
            # Pull audio from queue
            try:
                chunk = self.audio_q.get(timeout=0.15)
            except queue.Empty:
                # If we toggled off mid-speech, flush whatever we have
                if not self.is_active and triggered and speech_buf:
                    if speech_n >= MIN_SPEECH_FRAMES:
                        audio = np.concatenate(speech_buf)
                        text = self._transcribe(audio)
                        if text:
                            self._type_text(text)
                    speech_buf.clear()
                    triggered = False
                    silence_n = 0
                    speech_n = 0
                    ring = np.array([], dtype=np.float32)
                continue

            if not self.is_active:
                continue

            # Accumulate into ring buffer, process in fixed frames
            ring = np.concatenate([ring, chunk])

            while len(ring) >= FRAME_SAMPLES:
                frame_f32 = ring[:FRAME_SAMPLES]
                ring = ring[FRAME_SAMPLES:]

                # webrtcvad needs 16-bit PCM bytes
                pcm16 = (frame_f32 * 32767).astype(np.int16).tobytes()

                try:
                    is_speech = self.vad.is_speech(pcm16, SAMPLE_RATE)
                except Exception:
                    is_speech = False

                if is_speech:
                    speech_buf.append(frame_f32)
                    speech_n += 1
                    silence_n = 0
                    triggered = True

                    # Force-flush very long utterances
                    if speech_n >= MAX_SPEECH_FRAMES:
                        audio = np.concatenate(speech_buf)
                        text = self._transcribe(audio)
                        if text:
                            self._type_text(text)
                        speech_buf.clear()
                        speech_n = 0

                elif triggered:
                    speech_buf.append(frame_f32)   # keep trailing silence for context
                    silence_n += 1

                    if silence_n >= SILENCE_FRAMES:
                        # Silence long enough — transcribe
                        if speech_n >= MIN_SPEECH_FRAMES:
                            audio = np.concatenate(speech_buf)
                            text = self._transcribe(audio)
                            if text:
                                self._type_text(text)
                                cprint(
                                    f"\r{GREEN}{BOLD}● LISTENING{RESET}    "
                                    f"{DIM}» {text}{RESET}"
                                    + " " * 20,
                                    end="",
                                )

                        speech_buf.clear()
                        triggered = False
                        silence_n = 0
                        speech_n = 0

    # ── entry point ────────────────────────────────────
    def run(self):
        self.load_model()

        # Preflight: check typing tool
        tool = "wtype" if self._wayland else "xdotool"
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            cprint(f"{RED}✘  '{tool}' not found. Run: sudo apt install {tool}{RESET}")
            sys.exit(1)

        proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        proc_thread.start()

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=self._audio_cb,
        )

        listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

        cprint("")
        cprint(f"{BOLD}{'═' * 50}{RESET}")
        cprint(f"{BOLD}  Whisper Live Transcriber{RESET}")
        cprint(f"{DIM}  Model : {self.model_size} / {self.device} / {self.compute_type}{RESET}")
        cprint(f"{DIM}  Typing: {tool} ({'Wayland' if self._wayland else 'X11'}){RESET}")
        cprint(f"{DIM}  Toggle: Ctrl + Alt + S{RESET}")
        cprint(f"{DIM}  Quit  : Ctrl + C{RESET}")
        cprint(f"{BOLD}{'═' * 50}{RESET}")
        cprint(f"\r{RED}○ PAUSED{RESET}       {DIM}(Ctrl+Alt+S to start){RESET}   ", end="")

        with stream:
            listener.start()
            try:
                while True:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                cprint(f"\n\n{DIM}Shutting down…{RESET}")
                listener.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time speech-to-text at your cursor using Whisper."
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        choices=["tiny.en", "base.en", "small.en", "medium.en", "large-v3"],
        help=f"Whisper model size (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-d", "--device",
        default=DEFAULT_DEVICE,
        choices=["cuda", "cpu"],
        help=f"Inference device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "-c", "--compute-type",
        default=DEFAULT_COMPUTE,
        choices=["float16", "int8", "int8_float16", "float32"],
        help=f"Compute type for CTranslate2 (default: {DEFAULT_COMPUTE})",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=(
            f"Whisper language code, e.g. en, fr, de, zh (default: {DEFAULT_LANGUAGE}). "
            "For non-English use large-v3 (-m large-v3); .en models are English-only."
        ),
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=SILENCE_MS,
        help=f"Silence duration (ms) before transcribing (default: {SILENCE_MS})",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio input devices and exit",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Audio input device index (see --list-devices)",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if args.list_devices:
        cprint(str(sd.query_devices()))
        sys.exit(0)

    if args.input_device is not None:
        sd.default.device[0] = args.input_device

    # Update silence threshold if overridden
    global SILENCE_FRAMES
    SILENCE_FRAMES = int(args.silence_ms / FRAME_MS)

    transcriber = LiveTranscriber(args.model, args.device, args.compute_type, args.language)
    transcriber.run()


if __name__ == "__main__":
    main()
