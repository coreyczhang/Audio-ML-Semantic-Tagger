"""
MetaMusic Tagger
================
AI-powered music metadata tagging for sync licensing.
Uses librosa acoustic analysis + rule-based classification.
No API key required — runs 100% locally.

Usage:
    python metamusic_tagger.py                          # tag ./audio_files folder
    python metamusic_tagger.py /path/to/folder          # tag specific folder
    python metamusic_tagger.py /path/to/folder out.xlsx # custom output path

Requirements:
    pip install librosa soundfile numpy pandas openpyxl scipy

Author: Team X — Northwestern MetaMusic Project
"""

import os
import sys
import warnings
import time

import librosa
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Rule-based tag classifiers
# ---------------------------------------------------------------------------

def classify_genre(tempo, spectral_centroid, harmonic_ratio, onset_density,
                   rms, zcr, mfcc_1, key, mode):
    """
    Infer genre from acoustic features using threshold rules.
    Returns (primary_genre, subgenre).
    """
    genres = []

    # ── Electronic / Ambient / Lo-fi ──────────────────────────────────────
    is_electronic = spectral_centroid > 2500 and harmonic_ratio < 0.55
    is_ambient    = onset_density < 1.5 and rms < 0.05 and harmonic_ratio > 0.45
    is_lofi       = spectral_centroid < 1800 and tempo < 100 and rms < 0.06

    # ── Cinematic / Orchestral ─────────────────────────────────────────────
    is_cinematic  = harmonic_ratio > 0.65 and onset_density < 3.0 and rms > 0.03

    # ── Hip-Hop / Trap / R&B ───────────────────────────────────────────────
    is_hiphop     = 60 <= tempo <= 110 and harmonic_ratio < 0.55 and onset_density > 2.0
    is_trap       = 60 <= tempo <= 80  and harmonic_ratio < 0.45 and spectral_centroid > 2000

    # ── Pop / Indie Pop ───────────────────────────────────────────────────
    is_pop        = 100 <= tempo <= 145 and rms > 0.05 and harmonic_ratio > 0.45

    # ── Rock / Alternative ────────────────────────────────────────────────
    is_rock       = rms > 0.08 and zcr > 0.06 and onset_density > 3.0

    # ── Jazz / Soul / Blues ───────────────────────────────────────────────
    is_jazz       = harmonic_ratio > 0.60 and 60 <= tempo <= 180 and onset_density > 1.5 and rms < 0.10

    # ── Classical / Neoclassical ──────────────────────────────────────────
    is_classical  = harmonic_ratio > 0.70 and zcr < 0.04

    # ── Chill / Downtempo ─────────────────────────────────────────────────
    is_chill      = tempo < 95 and rms < 0.07 and onset_density < 3.5

    # Priority-ordered decision
    if is_trap:
        return "Hip-Hop", "Trap"
    if is_hiphop:
        return "Hip-Hop", "Lo-fi Hip-Hop" if is_lofi else "Hip-Hop / R&B"
    if is_classical:
        return "Classical", "Neoclassical" if spectral_centroid < 2500 else "Contemporary Classical"
    if is_cinematic and not is_electronic:
        return "Cinematic", "Orchestral / Cinematic" if harmonic_ratio > 0.70 else "Cinematic Electronic"
    if is_jazz:
        return "Jazz", "Nu-Jazz" if is_electronic else "Jazz / Soul"
    if is_rock:
        return "Rock", "Alternative Rock" if mode == "Minor" else "Indie Rock"
    if is_ambient and is_lofi:
        return "Electronic", "Lo-fi Ambient"
    if is_ambient:
        return "Electronic", "Ambient / Atmospheric"
    if is_lofi:
        return "Electronic", "Lo-fi Chill"
    if is_electronic:
        subgenre = "Synthwave" if mode == "Minor" else "Electronic Pop"
        return "Electronic", subgenre
    if is_pop:
        return "Pop", "Indie Pop" if rms < 0.09 else "Pop"
    if is_chill:
        return "Electronic", "Downtempo / Chillout"

    # Fallback
    return "Pop", "Contemporary"


def classify_mood(tempo, rms, mode, harmonic_ratio, onset_density,
                  spectral_centroid, zcr):
    """Return a list of 3 mood descriptors."""
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

    # Valence axis (major = positive, minor = darker)
    if mode == "Major":
        if tempo > 120:
            moods.append("Uplifting")
        elif tempo > 90:
            moods.append("Positive")
        else:
            moods.append("Nostalgic")
    else:  # Minor
        if tempo < 80:
            moods.append("Melancholic")
        elif rms > 0.08:
            moods.append("Tense")
        else:
            moods.append("Introspective")

    # Texture / feel axis
    if onset_density < 1.5 and harmonic_ratio > 0.5:
        moods.append("Dreamy")
    elif onset_density > 4.0:
        moods.append("Driving")
    elif spectral_centroid < 1500:
        moods.append("Warm")
    elif spectral_centroid > 3500:
        moods.append("Ethereal")
    elif zcr > 0.08:
        moods.append("Gritty")
    else:
        moods.append("Smooth")

    return moods[:3]


def classify_instrumentation(harmonic_ratio, spectral_centroid, zcr, rms,
                              onset_density, tempo, mfcc_1):
    """Infer likely instruments from spectral characteristics."""
    instruments = []

    is_mostly_harmonic  = harmonic_ratio > 0.60
    is_mostly_percussive = harmonic_ratio < 0.40
    is_bright           = spectral_centroid > 3000
    is_warm             = spectral_centroid < 1500
    is_mid              = 1500 <= spectral_centroid <= 3000
    is_noisy            = zcr > 0.08
    is_loud             = rms > 0.08
    is_sparse           = onset_density < 2.0
    is_busy             = onset_density > 4.0

    # Lead / melodic instruments
    if is_mostly_harmonic and is_bright:
        instruments.append("Synthesizer Lead")
    elif is_mostly_harmonic and is_warm:
        instruments.append("Electric Piano" if mfcc_1 > -200 else "Piano")
    elif is_mostly_harmonic and is_mid:
        instruments.append("Piano" if mfcc_1 < -100 else "Organ")

    # Pad / texture
    if is_sparse and harmonic_ratio > 0.50:
        instruments.append("Synth Pad")
    elif harmonic_ratio > 0.65 and not is_noisy:
        instruments.append("Strings" if spectral_centroid < 2500 else "Choir Synth")

    # Bass
    if is_warm and rms > 0.03:
        instruments.append("Bass Guitar" if zcr > 0.04 else "Sub Bass")

    # Rhythm / percussion
    if is_mostly_percussive or onset_density > 3.0:
        if is_noisy:
            instruments.append("Drum Kit")
        else:
            instruments.append("Electronic Drums" if spectral_centroid > 2500 else "Drum Machine")

    # Guitar (bright + harmonic + moderate zcr)
    if 0.04 < zcr < 0.08 and harmonic_ratio > 0.50 and is_mid:
        instruments.append("Acoustic Guitar" if is_warm else "Electric Guitar")

    if not instruments:
        instruments = ["Synthesizer", "Electronic Drums"]

    return instruments[:5]


def classify_vocals(zcr, spectral_centroid, harmonic_ratio, mfcc_2, rms):
    """
    Estimate vocal presence from spectral cues.
    Human voice typically lives in 300–3400 Hz range, creates
    specific MFCC and ZCR signatures.
    """
    # High ZCR + mid spectral centroid + low-moderate harmonic ratio → likely vocal presence
    vocal_score = 0
    if 0.05 < zcr < 0.15:
        vocal_score += 1
    if 1000 < spectral_centroid < 3500:
        vocal_score += 1
    if 0.35 < harmonic_ratio < 0.70:
        vocal_score += 1
    if -50 < mfcc_2 < 100:
        vocal_score += 1

    if vocal_score <= 1:
        return "No Vocals"
    elif vocal_score == 2:
        return "No Vocals"   # borderline — default to instrumental
    elif spectral_centroid > 2500:
        return "Female Vocal"
    elif spectral_centroid < 1800:
        return "Male Vocal"
    else:
        return "Vocal"


def classify_sync_use_cases(genre, mood_list, tempo, rms, mode, harmonic_ratio):
    """Suggest sync placement contexts."""
    use_cases = []
    mood_str = " ".join(mood_list).lower()
    genre_lower = genre.lower()

    # Cinematic / drama
    if harmonic_ratio > 0.60 and "melancholic" in mood_str or "tense" in mood_str:
        use_cases.append("Dramatic TV scene")
    if harmonic_ratio > 0.65 and rms < 0.06:
        use_cases.append("Film score / underscore")

    # Commercial
    if "uplifting" in mood_str or "positive" in mood_str:
        use_cases.append("Lifestyle brand commercial")
    if tempo > 120 and rms > 0.07:
        use_cases.append("Sports highlight reel")

    # Ambient / background
    if rms < 0.05 and "calm" in mood_str or "relaxed" in mood_str:
        use_cases.append("Podcast / YouTube background")
    if "dreamy" in mood_str or "smooth" in mood_str:
        use_cases.append("Travel / documentary")

    # Genre-specific
    if "hip-hop" in genre_lower:
        use_cases.append("Fashion / streetwear brand")
    if "electronic" in genre_lower and tempo > 110:
        use_cases.append("Action / gaming montage")
    if "jazz" in genre_lower or "soul" in genre_lower:
        use_cases.append("Café / restaurant ambiance")
    if "cinematic" in genre_lower:
        use_cases.append("Trailer / teaser")

    # Fallback
    if not use_cases:
        use_cases = ["Background music", "Social media content"]

    return use_cases[:4]


def generate_tags(genre, subgenre, mood_list, instruments, vocals,
                  tempo_feel, energy_level, mode, key):
    """Flatten everything into a searchable keyword list."""
    tags = set()
    tags.add(genre.lower())
    tags.add(subgenre.lower())
    tags.update([m.lower() for m in mood_list])
    tags.update([i.lower() for i in instruments])
    tags.add(vocals.lower())
    tags.add(tempo_feel.lower())
    tags.add(energy_level.lower())
    tags.add(mode.lower())
    tags.add(f"{key} {mode}".lower())

    # Remove very generic tokens
    tags.discard("no vocals")
    tags.discard("vocal")

    return sorted(tags)[:12]


# ---------------------------------------------------------------------------
# Main tagger class
# ---------------------------------------------------------------------------

class MetaMusicTagger:
    """
    Extract sync licensing metadata from audio files using:
    - Librosa acoustic feature extraction
    - Rule-based classification (no API key needed)
    """

    PITCH_CLASSES   = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    MAJOR_PROFILE   = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR_PROFILE   = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    # ------------------------------------------------------------------
    # Step 1 — Acoustic feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, audio_path: str, duration: float = 60.0) -> dict:
        """
        Extract acoustic features from an audio file using librosa.

        Args:
            audio_path: Path to audio file (.wav, .mp3, .flac, etc.)
            duration:   Max seconds to analyse (default 60 s)

        Returns:
            Dictionary of numeric acoustic measurements.
        """
        y, sr = librosa.load(audio_path, duration=duration, mono=True)

        # Tempo & beats
        tempo_result, beats = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo_result, np.ndarray):
            tempo = float(tempo_result.item() if tempo_result.size == 1 else tempo_result[0])
        else:
            tempo = float(tempo_result)

        # Key & mode (Krumhansl-Schmuckler)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        major_scores = [np.corrcoef(np.roll(chroma_mean, -i), self.MAJOR_PROFILE)[0, 1]
                        for i in range(12)]
        minor_scores = [np.corrcoef(np.roll(chroma_mean, -i), self.MINOR_PROFILE)[0, 1]
                        for i in range(12)]

        best_major = int(np.argmax(major_scores))
        best_minor = int(np.argmax(minor_scores))

        if max(major_scores) >= max(minor_scores):
            key  = self.PITCH_CLASSES[best_major]
            mode = "Major"
        else:
            key  = self.PITCH_CLASSES[best_minor]
            mode = "Minor"

        top_notes = [self.PITCH_CLASSES[i] for i in np.argsort(chroma_mean)[-3:][::-1]]

        # Energy
        rms = float(np.mean(librosa.feature.rms(y=y)))

        # Spectral features
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        zcr               = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # Harmonic vs. percussive ratio
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_e  = float(np.mean(librosa.feature.rms(y=y_harm)))
        perc_e  = float(np.mean(librosa.feature.rms(y=y_perc)))
        harmonic_ratio = harm_e / (harm_e + perc_e + 1e-8)

        # Onset density
        onset_frames   = librosa.onset.onset_detect(y=y, sr=sr)
        duration_actual = len(y) / sr
        onset_density   = len(onset_frames) / duration_actual if duration_actual > 0 else 0.0

        # MFCCs
        mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1).tolist()

        # Tempo feel label
        if tempo < 70:
            tempo_feel = "Slow"
        elif tempo < 100:
            tempo_feel = "Medium"
        elif tempo < 140:
            tempo_feel = "Upbeat"
        else:
            tempo_feel = "Fast"

        # Energy level label
        if rms < 0.03:
            energy_level = "Low"
        elif rms < 0.06:
            energy_level = "Medium"
        elif rms < 0.10:
            energy_level = "High"
        else:
            energy_level = "Very High"

        return {
            "tempo":             tempo,
            "tempo_feel":        tempo_feel,
            "key":               key,
            "mode":              mode,
            "top_notes":         ", ".join(top_notes),
            "rms":               rms,
            "energy_level":      energy_level,
            "spectral_centroid": spectral_centroid,
            "spectral_rolloff":  spectral_rolloff,
            "zcr":               zcr,
            "harmonic_ratio":    harmonic_ratio,
            "onset_density":     onset_density,
            "mfcc_1":            mfcc_means[0],
            "mfcc_2":            mfcc_means[1],
            "mfcc_3":            mfcc_means[2],
            "mfcc_4":            mfcc_means[3],
            "mfcc_5":            mfcc_means[4],
        }

    # ------------------------------------------------------------------
    # Step 2 — Rule-based tag generation
    # ------------------------------------------------------------------

    def interpret_features(self, features: dict) -> dict:
        """
        Apply rule-based classifiers to produce human-readable tags.

        Args:
            features: Output of extract_features()

        Returns:
            Dictionary with genre, mood, instrumentation, etc.
        """
        f = features  # shorthand

        genre, subgenre = classify_genre(
            tempo=f["tempo"],
            spectral_centroid=f["spectral_centroid"],
            harmonic_ratio=f["harmonic_ratio"],
            onset_density=f["onset_density"],
            rms=f["rms"],
            zcr=f["zcr"],
            mfcc_1=f["mfcc_1"],
            key=f["key"],
            mode=f["mode"],
        )

        mood_list = classify_mood(
            tempo=f["tempo"],
            rms=f["rms"],
            mode=f["mode"],
            harmonic_ratio=f["harmonic_ratio"],
            onset_density=f["onset_density"],
            spectral_centroid=f["spectral_centroid"],
            zcr=f["zcr"],
        )

        instruments = classify_instrumentation(
            harmonic_ratio=f["harmonic_ratio"],
            spectral_centroid=f["spectral_centroid"],
            zcr=f["zcr"],
            rms=f["rms"],
            onset_density=f["onset_density"],
            tempo=f["tempo"],
            mfcc_1=f["mfcc_1"],
        )

        vocals = classify_vocals(
            zcr=f["zcr"],
            spectral_centroid=f["spectral_centroid"],
            harmonic_ratio=f["harmonic_ratio"],
            mfcc_2=f["mfcc_2"],
            rms=f["rms"],
        )

        sync_use_cases = classify_sync_use_cases(
            genre=genre,
            mood_list=mood_list,
            tempo=f["tempo"],
            rms=f["rms"],
            mode=f["mode"],
            harmonic_ratio=f["harmonic_ratio"],
        )

        tags = generate_tags(
            genre=genre,
            subgenre=subgenre,
            mood_list=mood_list,
            instruments=instruments,
            vocals=vocals,
            tempo_feel=f["tempo_feel"],
            energy_level=f["energy_level"],
            mode=f["mode"],
            key=f["key"],
        )

        return {
            "genre":            [genre],
            "subgenre":         subgenre,
            "mood":             mood_list,
            "energy_level":     f["energy_level"],
            "tempo_feel":       f["tempo_feel"],
            "instrumentation":  instruments,
            "vocals":           vocals,
            "production_style": "electronic" if "Electronic" in genre else "acoustic",
            "sync_use_cases":   sync_use_cases,
            "tags":             tags,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag_file(self, audio_path: str) -> dict:
        """
        Analyse one audio file and return all metadata tags.
        """
        filename = os.path.basename(audio_path)
        print(f"  [extract]  {filename}")
        features = self.extract_features(audio_path)

        tags = self.interpret_features(features)
        print(f"  [tags]     {filename}  → {tags['genre'][0]} | {', '.join(tags['mood'])}")

        return {
            "filename":         filename,
            "tempo_bpm":        round(features["tempo"], 1),
            "key":              f"{features['key']} {features['mode']}",
            "energy_level":     tags["energy_level"],
            "tempo_feel":       tags["tempo_feel"],
            "genre":            " / ".join(tags["genre"]),
            "subgenre":         tags["subgenre"],
            "mood":             ", ".join(tags["mood"]),
            "instrumentation":  ", ".join(tags["instrumentation"]),
            "vocals":           tags["vocals"],
            "production_style": tags["production_style"],
            "sync_use_cases":   " | ".join(tags["sync_use_cases"]),
            "tags":             ", ".join(tags["tags"]),
        }

    def tag_folder(self, folder_path: str,
                   output_path: str = "metamusic_output.xlsx") -> pd.DataFrame:
        """
        Process every audio file in a folder and export results to Excel.
        """
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
            audio_path = os.path.join(folder_path, fname)
            try:
                row = self.tag_file(audio_path)
                rows.append(row)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                rows.append({"filename": fname, "error": str(e)})

        df = pd.DataFrame(rows)

        # Export
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_excel(output_path, index=False)
        print(f"\n{'=' * 60}")
        print(f"✓ Done!  Results saved to: {output_path}")
        print(f"  {len(df)} files processed")

        return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if len(args) == 0:
        folder_path = "audio_files"
        output_path = "metamusic_output.xlsx"
    elif len(args) == 1:
        folder_path = args[0]
        output_path = "metamusic_output.xlsx"
    elif len(args) == 2:
        folder_path = args[0]
        output_path = args[1]
    else:
        print("Usage: python metamusic_tagger.py [folder_path] [output_path.xlsx]")
        sys.exit(1)

    if not os.path.isdir(folder_path):
        print(f"Error: folder not found: {folder_path}")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          MetaMusic Tagger — Solo Hands Music LLC         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    tagger = MetaMusicTagger()
    tagger.tag_folder(folder_path, output_path)


if __name__ == "__main__":
    main()
