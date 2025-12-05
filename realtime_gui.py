

"""
===========================================================
                    Common Voice Gender Detection
===========================================================

Tkinter desktop app that:

- Records voice from the microphone
- Shows a LIVE waveform while recording
- Uses a trained CNN model to predict gender (male / female)
- Displays probabilities for both classes with progress bars

This file is only responsible for:
    • GUI (Tkinter + ttk)
    • Microphone recording (sounddevice)
    • Waveform drawing (matplotlib)
    • Loading the trained model (.h5)
    • Running prediction on the recorded audio

The model itself was trained in project.ipynb and saved as:
    best_gender_cnn.h5
===========================================================
"""

# ---------- Standard Library ----------
import os
import threading  # to run recording / prediction without freezing the GUI

# ---------- Third-party Libraries ----------
import numpy as np
import sounddevice as sd              # microphone recording
from scipy.io.wavfile import write    # save .wav files
import librosa                         # audio loading + MFCC

import tkinter as tk
from tkinter import ttk, messagebox    # Tkinter widgets + popup dialogs

from tensorflow.keras.models import load_model  # load trained .h5 model

# Matplotlib for drawing the waveform inside Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# =========================================================
#              MODEL / AUDIO CONFIG (MUST MATCH TRAINING)
# =========================================================

SR = 16000       # Sample rate used during training (16 kHz)
N_MFCC = 40      # Number of MFCC coefficients used
MAX_LEN = 44     # Number of time frames (columns) for MFCC
CHANNELS = 1     # Mono audio

# Base directory of this script (ML folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to trained model (same folder)
MODEL_PATH = os.path.join(BASE_DIR, "best_gender_cnn.h5")

# Temporary .wav file where we save last recording
TEMP_WAV = os.path.join(BASE_DIR, "mic_input.wav")

# =========================================================
#                 LOAD TRAINED KERAS MODEL
# =========================================================

model = None
model_load_error = None

try:
    # Try loading the model from disk
    model = load_model(MODEL_PATH)
except Exception as e:
    # Store error so we can show it in the GUI later
    model_load_error = str(e)
    print("Error loading model:", e)

# =========================================================
#          AUDIO -> MFCC FEATURE EXTRACTION FUNCTION
# =========================================================

def extract_mfcc(file_path, n_mfcc=N_MFCC, max_len=MAX_LEN, sr=SR):
    """
    Load audio from file, preprocess, and return normalized MFCC.

    Steps:
        1. Load audio using librosa (monophonic, sr=16000).
        2. Trim silence from beginning and end.
        3. Normalize waveform amplitude.
        4. Compute MFCC features (n_mfcc=40).
        5. Pad or crop along time dimension to fixed MAX_LEN=44.
        6. Standardize MFCC (mean 0, std 1).

    Returns:
        mfcc: np.ndarray with shape (n_mfcc, max_len)
    """
    # Load audio (librosa always returns float32 numpy array)
    y, sr = librosa.load(file_path, sr=sr)

    # Remove leading / trailing silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # Normalize audio (peak normalization)
    y = librosa.util.normalize(y)

    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Ensure fixed time length:
    # if too short, pad with zeros; if too long, crop.
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    # Standardize MFCC (better for neural networks)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    return mfcc

# =========================================================
#                     MAIN GUI CLASS
# =========================================================

class GenderGUI:
    """Main application class for the Voice Gender Detection GUI."""

    def __init__(self, root: tk.Tk):
        # ---- Basic window setup ----
        self.root = root
        self.root.title("Common Voice Gender Detection")
        self.root.geometry("900x520")
        self.root.configure(bg="#020617")  # dark background
        self.root.resizable(False, False)

        # ---- Variables used for audio streaming ----
        self.stream = None            # sounddevice.InputStream object
        self.is_recording = False     # flag to know if we are recording
        self.audio_chunks = []        # list of recorded chunks
        self.lock = threading.Lock()  # lock to protect audio_chunks in threads

        # ============================================
        #                 STYLES
        # ============================================

        style = ttk.Style()
        style.theme_use("clam")  # simple modern theme

        style.configure("Root.TFrame", background="#020617")
        style.configure("Card.TFrame", background="#020617")
        style.configure("Panel.TFrame", background="#020617")

        # Large heading text
        style.configure(
            "Title.TLabel",
            background="#020617",
            foreground="#f9fafb",
            font=("Segoe UI Semibold", 22),
        )

        # Slightly smaller grey text
        style.configure(
            "SubTitle.TLabel",
            background="#020617",
            foreground="#9ca3af",
            font=("Segoe UI", 11),
        )

        # Big prediction label ("male" / "female")
        style.configure(
            "BigResult.TLabel",
            background="#020617",
            foreground="#f97316",
            font=("Segoe UI Semibold", 28),
        )

        # Small, low-contrast informational text
        style.configure(
            "Muted.TLabel",
            background="#020617",
            foreground="#6b7280",
            font=("Segoe UI", 9),
        )

        # Accent style for buttons
        style.configure(
            "Accent.TButton",
            font=("Segoe UI Semibold", 12),
            padding=8,
        )

        # Progressbar style (orange bar)
        style.configure(
            "Orange.Horizontal.TProgressbar",
            troughcolor="#020617",
            bordercolor="#020617",
        )

        # ============================================
        #                 MAIN LAYOUT
        # ============================================

        # Outer container frame
        root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=20)
        root_frame.pack(fill="both", expand=True)

        # ---------- Header ----------
        header = ttk.Frame(root_frame, style="Root.TFrame")
        header.pack(fill="x", pady=(0, 15))

        ttk.Label(
            header,
            text="Common Voice Gender Detection",
            style="Title.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="Upload or record an audio clip to classify the speaker's gender as female or male.",
            style="SubTitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        # ---------- Two-column content area ----------
        content = ttk.Frame(root_frame, style="Root.TFrame")
        content.pack(fill="both", expand=True, pady=(10, 0))

        content.columnconfigure(0, weight=3)  # left panel (waveform)
        content.columnconfigure(1, weight=2)  # right panel (prediction)

        # =========================================================
        #                   LEFT PANEL (WAVEFORM)
        # =========================================================

        left_panel = ttk.Frame(content, style="Panel.TFrame")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        # Left panel title line (like “Upload Audio (WAV, ...)”)
        upload_header = ttk.Label(
            left_panel,
            text="♫  Microphone Input (16 kHz)",
            style="SubTitle.TLabel",
        )
        upload_header.pack(anchor="w", pady=(0, 6))

        # Frame that holds the waveform Matplotlib canvas
        waveform_frame = ttk.Frame(left_panel, style="Card.TFrame")
        waveform_frame.pack(fill="both", expand=True)

        # --------- Matplotlib Figure for waveform ---------
        self.fig = Figure(figsize=(5, 2.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#020617")

        # Configure axes: no ticks, minimal style
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            self.ax.spines[side].set_visible(False)

        # Line object that we'll update during recording
        self.wave_line, = self.ax.plot([], [], linewidth=1.2, color="#f97316")

        # Embed Matplotlib canvas into Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=waveform_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        # --------- Bottom controls: duration slider + buttons ---------
        controls = ttk.Frame(left_panel, style="Panel.TFrame")
        controls.pack(fill="x", pady=(10, 0))

        # Duration slider (2–8 seconds)
        self.duration_var = tk.IntVar(value=4)
        duration_frame = ttk.Frame(controls, style="Panel.TFrame")
        duration_frame.pack(side="left")

        ttk.Label(duration_frame, text="Duration:", style="Muted.TLabel").pack(
            side="left", padx=(0, 4)
        )

        self.duration_scale = ttk.Scale(
            duration_frame,
            from_=2,
            to=8,
            orient="horizontal",
            length=120,
            command=self._on_duration_change,  # called whenever slider moves
            value=4,
        )
        self.duration_scale.pack(side="left")

        self.duration_label = ttk.Label(
            duration_frame, text="4 sec", style="Muted.TLabel"
        )
        self.duration_label.pack(side="left", padx=(4, 0))

        # Clear button: reset waveform & prediction
        self.clear_button = ttk.Button(
            controls,
            text="Clear",
            style="Accent.TButton",
            command=self.clear_waveform,
        )
        self.clear_button.pack(side="right", padx=(6, 0))

        # Record button: toggles between Start/Stop
        self.record_button = ttk.Button(
            controls,
            text="🎙  Start Recording",
            style="Accent.TButton",
            command=self.toggle_recording,
        )
        self.record_button.pack(side="right")

        # =========================================================
        #                 RIGHT PANEL (PREDICTION)
        # =========================================================

        right_panel = ttk.Frame(content, style="Panel.TFrame")
        right_panel.grid(row=0, column=1, sticky="nsew")

        ttk.Label(
            right_panel,
            text="Gender Classification",
            style="SubTitle.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        result_card = ttk.Frame(right_panel, style="Card.TFrame")
        result_card.pack(fill="both", expand=True)

        # Big label that shows predicted gender
        self.pred_label = ttk.Label(
            result_card,
            text="—",
            style="BigResult.TLabel",
        )
        self.pred_label.pack(anchor="center", pady=(16, 6))

        # Progress bars for male / female probabilities
        bar_frame = ttk.Frame(result_card, style="Card.TFrame")
        bar_frame.pack(fill="x", padx=16, pady=(10, 0))

        # ---- Male row ----
        ttk.Label(bar_frame, text="male", style="SubTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.male_var = tk.DoubleVar(value=0.0)
        self.male_bar = ttk.Progressbar(
            bar_frame,
            variable=self.male_var,
            maximum=100,
            style="Orange.Horizontal.TProgressbar",
        )
        self.male_bar.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.male_pct_label = ttk.Label(
            bar_frame, text="0%", style="SubTitle.TLabel"
        )
        self.male_pct_label.grid(row=0, column=2, sticky="e", padx=(6, 0))

        # ---- Female row ----
        ttk.Label(bar_frame, text="female", style="SubTitle.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.female_var = tk.DoubleVar(value=0.0)
        self.female_bar = ttk.Progressbar(
            bar_frame,
            variable=self.female_var,
            maximum=100,
            style="Orange.Horizontal.TProgressbar",
        )
        self.female_bar.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))
        self.female_pct_label = ttk.Label(
            bar_frame, text="0%", style="SubTitle.TLabel"
        )
        self.female_pct_label.grid(row=1, column=2, sticky="e", padx=(6, 0), pady=(8, 0))

        # Make progressbars expand horizontally
        bar_frame.columnconfigure(1, weight=1)

        # ---- Status text under prediction ----
        self.status_var = tk.StringVar(
            value="Ready. Click “Start Recording” and speak into your microphone."
        )
        self.status_label = ttk.Label(
            right_panel, textvariable=self.status_var, style="Muted.TLabel"
        )
        self.status_label.pack(anchor="w", pady=(12, 0))

        # ---- Footer (model info) ----
        footer = ttk.Label(
            root_frame,
            text=f"Model: {os.path.basename(MODEL_PATH)}  •  Sample rate: 16 kHz",
            style="Muted.TLabel",
        )
        footer.pack(anchor="w", pady=(12, 0))

        # If model failed to load, disable recording and show error
        if model_load_error is not None:
            self.record_button.state(["disabled"])
            self.status_var.set("Error loading model. Check console output.")
            messagebox.showerror(
                "Model Error",
                f"Could not load model from:\n{MODEL_PATH}\n\nError:\n{model_load_error}",
            )

    # =========================================================
    #                     UI HELPER METHODS
    # =========================================================

    def _on_duration_change(self, val: str):
        """
        Called when the duration slider is moved.
        Updates the label text and IntVar.
        """
        secs = int(float(val))
        self.duration_var.set(secs)
        self.duration_label.config(text=f"{secs} sec")

    def clear_waveform(self):
        """
        Clear the waveform display and reset prediction values.
        """
        # Remove recorded audio in a thread-safe way
        with self.lock:
            self.audio_chunks = []

        # Reset waveform line
        self.wave_line.set_data([], [])
        self.ax.set_xlim(0, 1)
        self.canvas.draw_idle()

        # Reset prediction UI
        self.pred_label.config(text="—", foreground="#f97316")
        self.male_var.set(0.0)
        self.female_var.set(0.0)
        self.male_pct_label.config(text="0%")
        self.female_pct_label.config(text="0%")
        self.status_var.set("Cleared. Ready for a new recording.")

    # =========================================================
    #                 RECORDING CONTROL METHODS
    # =========================================================

    def toggle_recording(self):
        """
        Called when user clicks the main record button.
        Toggles between start_recording() and stop_recording().
        """
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """
        Start microphone recording using sounddevice.InputStream.

        Also:
            - clears previous audio
            - starts periodic waveform updates
            - schedules automatic stop after selected duration
        """
        if model is None:
            messagebox.showerror("Model Error", "Model is not loaded.")
            return

        try:
            # Clear previous audio
            with self.lock:
                self.audio_chunks = []

            duration = self.duration_var.get()

            # Update flags and UI
            self.is_recording = True
            self.record_button.config(text="⏹  Stop", state="normal")
            self.status_var.set(f"Recording... speak now ({duration} sec).")

            # Create an input stream that calls _audio_callback for each chunk
            self.stream = sd.InputStream(
                channels=1,
                samplerate=SR,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()

            # Start updating the waveform every 80 ms
            self._schedule_waveform_update()

            # Automatically stop after "duration" seconds
            self.root.after(duration * 1000, self.stop_recording)

        except Exception as e:
            self.is_recording = False
            self.record_button.config(text="🎙  Start Recording")
            messagebox.showerror("Audio Error", f"Could not start recording:\n{e}")

    def stop_recording(self):
        """
        Stop microphone recording if active.
        Then start a background thread to finalize and predict.
        """
        if not self.is_recording:
            return

        # Update flag and button text
        self.is_recording = False
        self.record_button.config(text="🎙  Start Recording")

        # Safely stop and close the audio stream
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception:
            pass

        # Run the slow part (saving + predicting) in another thread
        threading.Thread(target=self._finalize_and_predict, daemon=True).start()

    def _audio_callback(self, indata, frames, time, status):
        """
        Callback used by sounddevice.InputStream.
        This is called from a separate audio thread.

        indata: numpy array of shape (frames, channels)
        """
        if status:
            print(status)
        # Append a copy of the chunk to our list (thread-safe)
        with self.lock:
            self.audio_chunks.append(indata.copy())

    def _schedule_waveform_update(self):
        """
        Called once to start the periodic waveform updates.
        While recording is active, this method re-schedules itself every 80 ms.
        """
        if not self.is_recording:
            return
        self._update_waveform()
        self.root.after(80, self._schedule_waveform_update)

    def _update_waveform(self):
        """
        Redraws the waveform using the audio collected so far.
        """
        with self.lock:
            if not self.audio_chunks:
                return
            # Concatenate all small chunks into one long array
            data = np.concatenate(self.audio_chunks, axis=0).ravel()

        # Time axis in seconds
        t = np.linspace(0, len(data) / SR, num=len(data))

        # Update line data
        self.wave_line.set_data(t, data)

        # Adjust x limits to full duration
        if len(t) > 0:
            self.ax.set_xlim(0, t[-1])
        self.ax.set_ylim(-1.05, 1.05)

        # Redraw canvas
        self.canvas.draw_idle()

    # =========================================================
    #                   PREDICTION METHODS
    # =========================================================

    def _finalize_and_predict(self):
        """
        Combine audio chunks, save to TEMP_WAV,
        extract MFCC, run model prediction,
        and update the UI with results.
        """
        self.status_var.set("Processing audio and running model...")

        # Combine chunks into a single numpy array
        with self.lock:
            if not self.audio_chunks:
                self.status_var.set("No audio captured. Try again.")
                return
            audio = np.concatenate(self.audio_chunks, axis=0)

        # Save to .wav using SciPy
        write(TEMP_WAV, SR, audio)

        try:
            # Feature extraction
            mfcc = extract_mfcc(TEMP_WAV)
            x = mfcc.reshape(1, N_MFCC, MAX_LEN, CHANNELS)

            # Model predicts probability of FEMALE
            prob_female = float(model.predict(x)[0][0])
            prob_male = 1.0 - prob_female

            # Schedule UI update on the main thread
            self.root.after(
                0,
                lambda: self._update_prediction(prob_male, prob_female),
            )
        except Exception as e:
            # Show error message if anything fails
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Prediction Error", f"Could not run prediction:\n{e}"
                ),
            )
            self.root.after(
                0,
                lambda: self.status_var.set("Error during prediction. Try again."),
            )

    def _update_prediction(self, prob_male: float, prob_female: float):
        """
        Update the prediction text and progress bars in the UI.

        Args:
            prob_male:   probability of male (0–1)
            prob_female: probability of female (0–1)
        """
        male_pct = prob_male * 100
        female_pct = prob_female * 100

        # Decide label based on higher probability
        if male_pct >= female_pct:
            label = "male"
        else:
            label = "female"

        # Use bright orange for main label color
        color = "#f97316"
        self.pred_label.config(text=label, foreground=color)

        # Update bars and percentage labels
        self.male_var.set(male_pct)
        self.female_var.set(female_pct)

        self.male_pct_label.config(text=f"{male_pct:.0f}%")
        self.female_pct_label.config(text=f"{female_pct:.0f}%")

        self.status_var.set("Done. You can record again or clear the waveform.")

# =========================================================
#                      APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":
    # Create Tk root window and run the app
    root = tk.Tk()
    app = GenderGUI(root)
    root.mainloop()
