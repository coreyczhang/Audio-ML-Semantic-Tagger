"""
MetaMusic Tagger
================
Music metadata tagging for sync licensing.
Uses Essentia for feature extraction and classification,
librosa for MFCCs and mel-spectrogram visualization.
No LLM calls — runs 100% locally.

Usage:
    python metamusic_tagger.py                          # tag ./audio_files folder
    python metamusic_tagger.py /path/to/folder          # tag specific folder
    python metamusic_tagger.py /path/to/folder out.xlsx # custom output path

Requirements:
    pip install essentia librosa soundfile numpy pandas openpyxl scipy

Author: Team X — Northwestern MetaMusic Project
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore")

try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    print("ERROR: essentia not found. Install with: pip install essentia")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Essentia feature extraction
# ---------------------------------------------------------------------------

def extract_essentia_features(audio_path: str) -> dict:
    """
    Extract audio features using Essentia algorithms.
    Returns a dictionary of measurements.
    """
    # Load audio — Essentia uses float32, mono
    loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
    audio = loader()

    # --- Rhythm ---
    rhythm = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm(audio)

    # --- Key & mode ---
    key, scale, key_strength = es.KeyExtractor()(audio)

    # --- Energy & loudness ---
    energy     = float(es.Energy()(audio))
    rms        = float(np.sqrt(np.mean(audio ** 2)))
    loudness   = float(es.Loudness()(audio))

    # --- Danceability (0 = not danceable, 3 = very danceable) ---
    danceability, _ = es.Danceability()(audio)

    # --- Spectral centroid (brightness) ---
    spectral_centroid = float(es.SpectralCentroidTime()(audio))

    # --- Zero crossing rate ---
    zcr = float(es.ZeroCrossingRate()(audio))

    # --- MFCCs (frame-by-frame, then average) ---
    window    = es.Windowing(type='hann')
    spectrum  = es.Spectrum()
    mfcc_algo = es.MFCC(numberCoefficients=13)

    mfcc_frames = []
    for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512, startFromZero=True):
        spec = spectrum(window(frame))
        _, mfcc_frame = mfcc_algo(spec)
        mfcc_frames.append(mfcc_frame)

    mfcc_mean = np.mean(mfcc_frames, axis=0) if mfcc_frames else np.zeros(13)

    # --- Harmonic / percussive ratio via librosa HPSS ---
    # Resample to 22050 for librosa
    audio_lr = librosa.resample(audio.astype(np.float32), orig_sr=44100, target_sr=22050)
    y_harm, y_perc = librosa.effects.hpss(audio_lr)
    harm_rms = float(np.mean(librosa.feature.rms(y=y_harm)))
    perc_rms = float(np.mean(librosa.feature.rms(y=y_perc)))
    harmonic_ratio = harm_rms / (harm_rms + perc_rms + 1e-8)

    # --- Onset density ---
    onset_frames  = librosa.onset.onset_detect(y=audio_lr, sr=22050)
    duration      = len(audio) / 44100
    onset_density = len(onset_frames) / duration if duration > 0 else 0.0

    # --- Chroma (pitch class distribution) via librosa ---
    chroma     = librosa.feature.chroma_cqt(y=audio_lr, sr=22050)
    chroma_mean = np.mean(chroma, axis=1)
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    top_notes  = [pitch_classes[i] for i in np.argsort(chroma_mean)[-3:][::-1]]

    # --- Tempo feel label ---
    tempo = float(bpm)
    if tempo < 70:
        tempo_feel = "Slow"
    elif tempo < 100:
        tempo_feel = "Medium"
    elif tempo < 140:
        tempo_feel = "Upbeat"
    else:
        tempo_feel = "Fast"

    # --- Energy level label ---
    if rms < 0.03:
        energy_level = "Low"
    elif rms < 0.06:
        energy_level = "Medium"
    elif rms < 0.10:
        energy_level = "High"
    else:
        energy_level = "Very High"

    return {
        "tempo":            tempo,
        "tempo_feel":       tempo_feel,
        "key":              key,
        "mode":             scale,          # "major" or "minor" from Essentia
        "key_strength":     float(key_strength),
        "top_notes":        ", ".join(top_notes),
        "rms":              rms,
        "energy":           energy,
        "loudness":         loudness,
        "danceability":     float(danceability),
        "energy_level":     energy_level,
        "spectral_centroid":spectral_centroid,
        "zcr":              zcr,
        "harmonic_ratio":   harmonic_ratio,
        "onset_density":    onset_density,
        "mfcc":             mfcc_mean,      # numpy array, stored separately
    }


# ---------------------------------------------------------------------------
# Classification using Essentia features
# ---------------------------------------------------------------------------

def classify_genre(f: dict) -> tuple:
    """Infer genre from Essentia features. Returns (genre, subgenre)."""
    tempo   = f["tempo"]
    sc      = f["spectral_centroid"]
    hr      = f["harmonic_ratio"]
    od      = f["onset_density"]
    rms     = f["rms"]
    zcr     = f["zcr"]
    dance   = f["danceability"]
    mode    = f["mode"]

    # Electronic / Ambient / Lo-fi
    is_electronic  = sc > 2500 and hr < 0.55
    is_ambient     = od < 1.5 and rms < 0.05 and hr > 0.45
    is_lofi        = sc < 1800 and tempo < 100 and rms < 0.06

    # Cinematic / Orchestral
    is_cinematic   = hr > 0.65 and od < 3.0

    # Hip-Hop / Trap
    is_hiphop      = 60 <= tempo <= 110 and hr < 0.55 and od > 2.0
    is_trap        = 60 <= tempo <= 80  and hr < 0.45 and sc > 2000

    # Pop
    is_pop         = 100 <= tempo <= 145 and rms > 0.05 and hr > 0.45

    # Rock
    is_rock        = rms > 0.08 and zcr > 0.06 and od > 3.0

    # Jazz / Soul (high harmonic, high danceability, not too loud)
    is_jazz        = hr > 0.60 and 60 <= tempo <= 180 and od > 1.5 and rms < 0.10

    # Classical / Neoclassical
    is_classical   = hr > 0.70 and zcr < 0.04

    # Danceable Electronic
    is_dance       = dance > 1.5 and sc > 2000

    if is_trap:
        return "Hip-Hop", "Trap"
    if is_hiphop:
        return "Hip-Hop", "Lo-fi Hip-Hop" if is_lofi else "Hip-Hop / R&B"
    if is_classical:
        return "Classical", "Neoclassical" if sc < 2500 else "Contemporary Classical"
    if is_cinematic and not is_electronic:
        return "Cinematic", "Orchestral / Cinematic" if hr > 0.70 else "Cinematic Electronic"
    if is_jazz:
        return "Jazz", "Nu-Jazz" if is_electronic else "Jazz / Soul"
    if is_rock:
        return "Rock", "Alternative Rock" if mode == "minor" else "Indie Rock"
    if is_dance:
        return "Electronic", "Dance / Club"
    if is_ambient and is_lofi:
        return "Electronic", "Lo-fi Ambient"
    if is_ambient:
        return "Electronic", "Ambient / Atmospheric"
    if is_lofi:
        return "Electronic", "Lo-fi Chill"
    if is_electronic:
        return "Electronic", "Synthwave" if mode == "minor" else "Electronic Pop"
    if is_pop:
        return "Pop", "Indie Pop" if rms < 0.09 else "Pop"
    return "Electronic", "Downtempo / Chillout"


def classify_mood(f: dict) -> list:
    """Return 3 mood descriptors using Essentia danceability + key + energy."""
    tempo  = f["tempo"]
    rms    = f["rms"]
    mode   = f["mode"]
    dance  = f["danceability"]
    od     = f["onset_density"]
    sc     = f["spectral_centroid"]
    zcr    = f["zcr"]
    hr     = f["harmonic_ratio"]

    moods = []

    # Energy axis
    if rms > 0.10:
        moods.append("Energetic")
    elif rms > 0.06:
        moods.append("Dynamic")
    elif rms > 0.03:
        moods.append("Relaxed")
    else:
        moods.append("Calm")

    # Valence axis (Essentia key scale: "major" / "minor")
    if mode == "major":
        moods.append("Uplifting" if tempo > 120 else "Positive" if tempo > 90 else "Nostalgic")
    else:
        moods.append("Melancholic" if tempo < 80 else "Tense" if rms > 0.08 else "Introspective")

    # Texture / feel — use Essentia danceability
    if dance > 1.8:
        moods.append("Groovy")
    elif od < 1.5 and hr > 0.5:
        moods.append("Dreamy")
    elif od > 4.0:
        moods.append("Driving")
    elif sc < 1500:
        moods.append("Warm")
    elif sc > 3500:
        moods.append("Ethereal")
    elif zcr > 0.08:
        moods.append("Gritty")
    else:
        moods.append("Smooth")

    return moods[:3]


def classify_instrumentation(f: dict) -> list:
    """Infer instruments from spectral characteristics."""
    hr   = f["harmonic_ratio"]
    sc   = f["spectral_centroid"]
    zcr  = f["zcr"]
    rms  = f["rms"]
    od   = f["onset_density"]
    m1   = f["mfcc"][0]

    instruments = []

    if hr > 0.60 and sc > 3000:
        instruments.append("Synthesizer Lead")
    elif hr > 0.60 and sc < 1500:
        instruments.append("Electric Piano" if m1 > -200 else "Piano")
    elif hr > 0.60:
        instruments.append("Piano" if m1 < -100 else "Organ")

    if od < 2.0 and hr > 0.50:
        instruments.append("Synth Pad")
    elif hr > 0.65 and zcr < 0.04:
        instruments.append("Strings" if sc < 2500 else "Choir Synth")

    if sc < 1500 and rms > 0.03:
        instruments.append("Bass Guitar" if zcr > 0.04 else "Sub Bass")

    if hr < 0.45 or od > 3.0:
        instruments.append("Drum Kit" if zcr > 0.06 else "Electronic Drums" if sc > 2500 else "Drum Machine")

    if 0.04 < zcr < 0.08 and hr > 0.50 and 1500 <= sc <= 3000:
        instruments.append("Acoustic Guitar" if sc < 2000 else "Electric Guitar")

    return (instruments or ["Synthesizer", "Electronic Drums"])[:5]


def classify_vocals(f: dict) -> str:
    """Estimate vocal presence from spectral signature."""
    zcr = f["zcr"]
    sc  = f["spectral_centroid"]
    hr  = f["harmonic_ratio"]
    m2  = f["mfcc"][1]

    score = sum([
        0.05 < zcr < 0.15,
        1000 < sc < 3500,
        0.35 < hr < 0.70,
        -50 < m2 < 100,
    ])

    if score <= 2:
        return "No Vocals"
    return "Female Vocal" if sc > 2500 else "Male Vocal"


def classify_sync_use_cases(genre: str, mood_list: list, f: dict) -> list:
    """Suggest sync placement contexts."""
    tempo      = f["tempo"]
    rms        = f["rms"]
    hr         = f["harmonic_ratio"]
    dance      = f["danceability"]
    mood_str   = " ".join(mood_list).lower()
    genre_lower = genre.lower()

    uses = []
    if hr > 0.60 and ("melancholic" in mood_str or "tense" in mood_str):
        uses.append("Dramatic TV scene")
    if hr > 0.65 and rms < 0.06:
        uses.append("Film score / underscore")
    if "uplifting" in mood_str or "positive" in mood_str:
        uses.append("Lifestyle brand commercial")
    if tempo > 120 and rms > 0.07:
        uses.append("Sports highlight reel")
    if dance > 1.5:
        uses.append("Fashion / party montage")
    if rms < 0.05 and ("calm" in mood_str or "relaxed" in mood_str):
        uses.append("Podcast / YouTube background")
    if "dreamy" in mood_str or "smooth" in mood_str:
        uses.append("Travel / documentary")
    if "hip-hop" in genre_lower:
        uses.append("Fashion / streetwear brand")
    if "electronic" in genre_lower and tempo > 110:
        uses.append("Action / gaming montage")
    if "jazz" in genre_lower:
        uses.append("Café / restaurant ambiance")
    if "cinematic" in genre_lower:
        uses.append("Trailer / teaser")

    return (uses or ["Background music", "Social media content"])[:4]


def generate_tags(genre, subgenre, mood_list, instruments, vocals,
                  tempo_feel, energy_level, mode, key, danceability) -> list:
    """Flatten everything into a searchable keyword list."""
    tags = set()
    tags.add(genre.lower())
    tags.add(subgenre.lower())
    tags.update(m.lower() for m in mood_list)
    tags.update(i.lower() for i in instruments)
    tags.add(tempo_feel.lower())
    tags.add(energy_level.lower())
    tags.add(mode.lower())
    tags.add(f"{key} {mode}".lower())
    if danceability > 1.5:
        tags.add("danceable")
    tags.discard("no vocals")
    return sorted(tags)[:12]


# ---------------------------------------------------------------------------
# Main tagger class
# ---------------------------------------------------------------------------

class MetaMusicTagger:
    """
    Extract sync licensing metadata using Essentia + librosa.
    No LLM calls — fully local.
    """

    def tag_file(self, audio_path: str) -> dict:
        filename = os.path.basename(audio_path)
        print(f"  [essentia] {filename}")

        f = extract_essentia_features(audio_path)

        genre, subgenre   = classify_genre(f)
        mood_list         = classify_mood(f)
        instruments       = classify_instrumentation(f)
        vocals            = classify_vocals(f)
        sync_use_cases    = classify_sync_use_cases(genre, mood_list, f)
        tags              = generate_tags(
            genre, subgenre, mood_list, instruments, vocals,
            f["tempo_feel"], f["energy_level"], f["mode"], f["key"], f["danceability"]
        )

        print(f"  [tags]     {filename} → {genre} | {', '.join(mood_list)}")

        return {
            "filename":         filename,
            "tempo_bpm":        round(f["tempo"], 1),
            "key":              f"{f['key']} {f['mode'].capitalize()}",
            "key_strength":     round(f["key_strength"], 2),
            "danceability":     round(f["danceability"], 2),
            "energy_level":     f["energy_level"],
            "tempo_feel":       f["tempo_feel"],
            "genre":            genre,
            "subgenre":         subgenre,
            "mood":             ", ".join(mood_list),
            "instrumentation":  ", ".join(instruments),
            "vocals":           vocals,
            "sync_use_cases":   " | ".join(sync_use_cases),
            "tags":             ", ".join(tags),
        }

    def tag_folder(self, folder_path: str,
                   output_path: str = "metamusic_output.xlsx") -> pd.DataFrame:
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
        files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in audio_extensions
        ])

        if not files:
            print(f"No audio files found in: {folder_path}")
            return pd.DataFrame()

        print(f"\nMetaMusic Tagger — {len(files)} files from: {folder_path}")
        print("=" * 60)

        rows = []
        for i, fname in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {fname}")
            try:
                row = self.tag_file(os.path.join(folder_path, fname))
                rows.append(row)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                rows.append({"filename": fname, "error": str(e)})

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_excel(output_path, index=False)

        print(f"\n{'=' * 60}")
        print(f"✓ Done! Results saved to: {output_path}")
        print(f"  {len(df)} files processed")
        return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    folder_path = args[0] if len(args) >= 1 else "audio_files"
    output_path = args[1] if len(args) >= 2 else "metamusic_output.xlsx"

    if not os.path.isdir(folder_path):
        print(f"Error: folder not found: {folder_path}")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          MetaMusic Tagger — Solo Hands Music LLC         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    MetaMusicTagger().tag_folder(folder_path, output_path)


if __name__ == "__main__":
    main()
