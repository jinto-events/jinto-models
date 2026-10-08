"""Grille de référence d'un fichier audio : beat_this final0 (PyTorch, post-traitement « minimal »), en millisecondes.

Sert de référence aux consommateurs des fichiers ONNX (leur grille doit s'en tenir à 20 ms près). Le fichier est d'abord
décodé en mono 22 050 Hz par ffmpeg ; pour comparer ce que lit un consommateur, donner le fichier tel qu'il l'analyse.

Usage : ``uv run analyze.py <checkpoint final0.ckpt> <fichier audio>...`` ; une ligne JSON par fichier.
"""

import json
import subprocess
import sys

import numpy as np

from beat_this.inference import Audio2Beats

SAMPLE_RATE = 22_050


def main() -> None:
    tracker = Audio2Beats(sys.argv[1], "cpu")
    for path in sys.argv[2:]:
        pcm = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
            check=True,
            capture_output=True,
        ).stdout
        beats, bar_starts = tracker(np.frombuffer(pcm, dtype=np.float32).astype(np.float64), SAMPLE_RATE)
        print(
            json.dumps(
                {
                    "file": path,
                    "beatsMs": [round(float(t) * 1000) for t in beats],
                    "barStartsMs": [round(float(t) * 1000) for t in bar_starts],
                },
                separators=(",", ":"),
            )
        )


if __name__ == "__main__":
    main()
