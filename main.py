import tkinter as tk
from queue import Queue, Empty
from threading import Thread
import numpy as np
import pyaudio
from vosk import Model, KaldiRecognizer
import json
import os
import collections

try:
    from scipy.signal import butter, sosfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False

import time

# === AudioInputManager ===
class AudioInputManager:
    def __init__(self, sample_rate=16000, chunk_size=2048, channels=1, format_=pyaudio.paInt16):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format_
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = Queue()
        self.is_recording = False
        self.recording_thread = None

    def start(self):
        self.stream = self.audio.open(format=self.format, channels=self.channels,
                                      rate=self.sample_rate, input=True,
                                      frames_per_buffer=self.chunk_size)
        self.is_recording = True
        self.recording_thread = Thread(target=self._record, daemon=True)
        self.recording_thread.start()

    def _record(self):
        while self.is_recording:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            self.audio_queue.put(data)

    def get_audio_chunk(self, timeout=0.1):
        try:
            return self.audio_queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


# === NoiseFilter ===
class NoiseFilter:
    """RMS-based noise gate + bandpass filter (300 Hz - 3400 Hz)."""

    def __init__(self, sample_rate=16000, rms_threshold=50, low_freq=300, high_freq=3400):
        self.sample_rate = sample_rate
        self.rms_threshold = rms_threshold
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.enabled = True

        if SCIPY_AVAILABLE:
            nyq = sample_rate / 2.0
            low = low_freq / nyq
            high = high_freq / nyq
            self.sos = butter(4, [low, high], btype="band", output="sos")
        else:
            self.sos = None

    def filter(self, audio_chunk_bytes):
        """Apply bandpass filter and RMS noise gate.

        Returns filtered bytes, or None if the chunk is below the noise threshold.
        """
        if not self.enabled:
            return audio_chunk_bytes

        # Convert bytes -> int16 numpy array
        audio_np = np.frombuffer(audio_chunk_bytes, dtype=np.int16).astype(np.float32)

        # RMS noise gate on raw audio (before filtering)
        rms = float(np.sqrt(np.mean(audio_np ** 2)))
        if rms < self.rms_threshold:
            return None

        # Bandpass filter
        if self.sos is not None:
            audio_np = sosfilt(self.sos, audio_np)
        else:
            fft = np.fft.rfft(audio_np)
            freqs = np.fft.rfftfreq(len(audio_np), d=1.0 / self.sample_rate)
            fft[(freqs < self.low_freq) | (freqs > self.high_freq)] = 0
            audio_np = np.fft.irfft(fft, n=len(audio_np))

        # Convert back to int16 bytes
        filtered = np.clip(audio_np, -32768, 32767).astype(np.int16)
        return filtered.tobytes()


# === RollingWordBuffer ===
class RollingWordBuffer:
    """Keeps the last *maxlen* recognised words in a rolling deque."""

    def __init__(self, maxlen=10):
        self._buf = collections.deque(maxlen=maxlen)

    def add(self, word: str):
        if word.strip():
            self._buf.append(word.strip().lower())

    def get_all(self) -> list:
        return list(self._buf)

    def clear(self):
        self._buf.clear()


# === KeywordDetector ===
class KeywordDetector:
    """Detects safety-related keywords in recognised text."""

    DEFAULT_KEYWORDS = ["fire", "help", "danger", "stop", "alarm"]

    def __init__(self, keywords=None):
        base = list(self.DEFAULT_KEYWORDS)
        if keywords:
            base.extend(keywords)
        self.keywords = [kw.lower() for kw in base]

    def add_keyword(self, keyword: str):
        kw = keyword.strip().lower()
        if kw and kw not in self.keywords:
            self.keywords.append(kw)

    def detect(self, text: str) -> list:
        """Return a list of keywords found in *text* (case-insensitive)."""
        lower_text = text.lower()
        return [kw for kw in self.keywords if kw in lower_text]


# === HapticController ===
class HapticController:
    """Controls a haptic motor via Raspberry Pi GPIO or prints to console."""

    HAPTIC_PIN = 18

    def __init__(self):
        self._gpio_ok = False
        if RPI_AVAILABLE:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.HAPTIC_PIN, GPIO.OUT)
                self._pwm = GPIO.PWM(self.HAPTIC_PIN, 100)
                self._gpio_ok = True
            except Exception as exc:
                print(f"[HapticController] GPIO init failed: {exc}")

    def buzz(self, duration=0.3):
        """Single non-blocking buzz."""
        Thread(target=self._do_buzz, args=(duration,), daemon=True).start()

    def _do_buzz(self, duration):
        if self._gpio_ok:
            self._pwm.start(50)
            time.sleep(duration)
            self._pwm.stop()
        else:
            print(f"[HAPTIC] buzz {duration}s")

    def alert_pattern(self):
        """Three short buzzes (SOS-like pattern) triggered on keyword detection."""
        Thread(target=self._do_alert, daemon=True).start()

    def _do_alert(self):
        for _ in range(3):
            self._do_buzz(0.15)
            time.sleep(0.1)

    def cleanup(self):
        if self._gpio_ok:
            self._pwm.stop()
            GPIO.cleanup()


# === BrailleManager ===
class BrailleManager:
    def __init__(self, canvas_widget, status_label, words_per_block=10):
        self.canvas = canvas_widget
        self.status_label = status_label
        self.words_per_block = words_per_block
        self.all_words = []
        self.current_index = 0

        self.braille_dict = {
            "a": [1], "b": [1, 2], "c": [1, 4], "d": [1, 4, 5], "e": [1, 5],
            "f": [1, 2, 4], "g": [1, 2, 4, 5], "h": [1, 2, 5], "i": [2, 4],
            "j": [2, 4, 5], "k": [1, 3], "l": [1, 2, 3], "m": [1, 3, 4],
            "n": [1, 3, 4, 5], "o": [1, 3, 5], "p": [1, 2, 3, 4], "q": [1, 2, 3, 4, 5],
            "r": [1, 2, 3, 5], "s": [2, 3, 4], "t": [2, 3, 4, 5], "u": [1, 3, 6],
            "v": [1, 2, 3, 6], "w": [2, 4, 5, 6], "x": [1, 3, 4, 6],
            "y": [1, 3, 4, 5, 6],
            "z": [1, 3, 5, 6],
            "0": [3, 4, 5, 6], "1": [1], "2": [1, 2], "3": [1, 4], "4": [1, 4, 5],
            "5": [1, 5], "6": [1, 2, 4], "7": [1, 2, 4, 5], "8": [1, 2, 5], "9": [2, 4],
            " ": [], ".": [2, 3, 4, 5, 6], ",": [2], "?": [2, 3, 4, 5], "!": [2, 3, 5],
            "'": [3], "-": [3, 6], "(": [1, 2, 3, 5, 6], ")": [2, 3, 4, 5, 6],
            ":": [1, 5, 6], ";": [1, 4, 5, 6], "&": [1, 2, 3, 4, 6],
        }

    def draw_braille_cell_direct(self, x_start, y_start, width, height, active_dots):
        center_x = x_start + width // 2
        center_y = y_start + height // 2
        dot_radius = 5
        col_spacing = 16
        row_spacing = 14
        positions = {
            1: (center_x - col_spacing // 2, center_y - row_spacing),
            2: (center_x - col_spacing // 2, center_y),
            3: (center_x - col_spacing // 2, center_y + row_spacing),
            4: (center_x + col_spacing // 2, center_y - row_spacing),
            5: (center_x + col_spacing // 2, center_y),
            6: (center_x + col_spacing // 2, center_y + row_spacing),
        }
        for i in range(1, 7):
            x, y = positions[i]
            color = "black" if i in active_dots else "lightgray"
            border_color = "black" if i in active_dots else "gray"
            self.canvas.create_oval(x - dot_radius, y - dot_radius,
                                    x + dot_radius, y + dot_radius,
                                    fill=color, outline=border_color, width=1)

    def show_braille_block(self):
        self.canvas.delete("all")
        if not self.all_words:
            self.status_label.config(text="No words yet. Speak!", fg="red")
            return
        block_num = (self.current_index // self.words_per_block) + 1
        total_blocks = (len(self.all_words) + self.words_per_block - 1) // self.words_per_block
        self.status_label.config(
            text=f"Words spoken: {len(self.all_words)} | Block: {block_num}/{total_blocks}", fg="green")
        block_words = self.all_words[self.current_index:self.current_index + self.words_per_block]
        text = " ".join(block_words)
        cell_width = 60
        cell_height = 60
        padding = 10
        x_pos = padding
        y_pos = padding
        for idx, letter in enumerate(text):
            self.canvas.create_rectangle(x_pos, y_pos, x_pos + cell_width, y_pos + cell_height,
                                         fill="white", outline="lightgray", width=1)
            active = self.braille_dict.get(letter.lower(), [])
            self.draw_braille_cell_direct(x_pos + 4, y_pos + 4, cell_width - 8, cell_height - 14, active)
            self.canvas.create_text(x_pos + cell_width // 2, y_pos + cell_height - 10,
                                    text=letter.upper(), font=("Arial", 10, "bold"), fill="darkgray")
            x_pos += cell_width + padding
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.canvas.xview_moveto(0)

    def next_block(self):
        if self.current_index + self.words_per_block < len(self.all_words):
            self.current_index += self.words_per_block
            self.show_braille_block()
        else:
            self.status_label.config(text="Last block!", fg="orange")

    def prev_block(self):
        if self.current_index >= self.words_per_block:
            self.current_index -= self.words_per_block
            self.show_braille_block()
        else:
            self.status_label.config(text="First block!", fg="orange")

    def clear_all(self):
        self.all_words.clear()
        self.current_index = 0
        self.canvas.delete("all")
        self.status_label.config(text="Cleared! Start speaking again.", fg="blue")

    def add_words(self, words):
        for word in words:
            if word.strip():
                self.all_words.append(word.lower())
        self.show_braille_block()


# === Tkinter Setup ===
root = tk.Tk()
root.title("Braille Simulation - English")
root.geometry("1200x650")

# Status label
status_label = tk.Label(root, text="Listening... Speak!", font=("Arial", 14), fg="blue", bg="lightgray")
status_label.pack(pady=10, fill="x")

# Keyword alert label
keyword_label = tk.Label(root, text="", font=("Arial", 12, "bold"), fg="red", bg="lightyellow")
keyword_label.pack(fill="x", padx=10)

# Canvas Frame
canvas_frame = tk.Frame(root, bg="white")
canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Scrollbar (horizontal)
scrollbar = tk.Scrollbar(canvas_frame, orient="horizontal")
scrollbar.pack(side="bottom", fill="x")

# Canvas (with scroll)
display_canvas = tk.Canvas(canvas_frame, bg="white", relief="sunken", bd=2, height=400,
                            xscrollcommand=scrollbar.set)
display_canvas.pack(side="top", fill="both", expand=True)

# Connect scrollbar to canvas
scrollbar.config(command=display_canvas.xview)

# === Create Braille Manager ===
braille_manager = BrailleManager(display_canvas, status_label, words_per_block=10)

# === Create pipeline components ===
noise_filter = NoiseFilter(sample_rate=16000, rms_threshold=50)
rolling_buffer = RollingWordBuffer(maxlen=10)
keyword_detector = KeywordDetector()
haptic_controller = HapticController()

# === VOSK setup ===
if not os.path.exists("vosk-model-small-en-us-0.15"):
    print("Vosk model 'vosk-model-small-en-us-0.15' not found. Download it from https://alphacephei.com/vosk/models")
    status_label.config(text="ERROR: Vosk model not found!", fg="red")
    exit(1)

model = Model("vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

# === Audio Manager ===
audio_manager = AudioInputManager()
audio_manager.start()


# === Real-time audio processing pipeline ===
def process_audio_loop():
    chunk = audio_manager.get_audio_chunk(timeout=0.1)
    if chunk:
        # 1. Noise filter
        filtered = noise_filter.filter(chunk)
        if filtered is None:
            root.after(100, process_audio_loop)
            return

        # 2. Speech-to-text
        if recognizer.AcceptWaveform(filtered):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                print(f"Recognised: {text}")

                # 3. Keyword detection
                found_keywords = keyword_detector.detect(text)
                if found_keywords:
                    kw_text = "⚠ KEYWORD: " + ", ".join(found_keywords).upper()
                    keyword_label.config(text=kw_text)
                    haptic_controller.alert_pattern()
                    print(f"[KeywordDetector] Found: {found_keywords}")

                # 4. Rolling buffer + Braille display
                words = text.split()
                for word in words:
                    rolling_buffer.add(word)
                braille_manager.add_words(words)

    root.after(100, process_audio_loop)

process_audio_loop()


# === Noise filter toggle ===
def toggle_noise_filter():
    noise_filter.enabled = not noise_filter.enabled
    state = "ON" if noise_filter.enabled else "OFF"
    btn_noise_filter.config(text=f"Noise Filter: {state}",
                            bg="lightgreen" if noise_filter.enabled else "lightyellow")

# === UI buttons ===
button_frame = tk.Frame(root, bg="lightgray")
button_frame.pack(pady=10, fill="x")

btn_prev = tk.Button(button_frame, text="← PREVIOUS", command=braille_manager.prev_block,
                     font=("Arial", 12), padx=20, pady=8, bg="lightblue")
btn_prev.pack(side="left", padx=10)

btn_next = tk.Button(button_frame, text="NEXT →", command=braille_manager.next_block,
                     font=("Arial", 12), padx=20, pady=8, bg="lightblue")
btn_next.pack(side="left", padx=10)

btn_clear = tk.Button(button_frame, text="CLEAR", command=braille_manager.clear_all,
                      font=("Arial", 12), padx=20, pady=8, bg="lightcoral")
btn_clear.pack(side="left", padx=10)

btn_noise_filter = tk.Button(button_frame, text="Noise Filter: ON", command=toggle_noise_filter,
                              font=("Arial", 12), padx=20, pady=8, bg="lightgreen")
btn_noise_filter.pack(side="left", padx=10)

root.protocol("WM_DELETE_WINDOW", lambda: (haptic_controller.cleanup(), audio_manager.stop(), root.destroy()))
root.mainloop()