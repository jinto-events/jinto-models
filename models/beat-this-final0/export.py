"""Export ONNX de beat_this final0 (CPJKU, MIT) : le modèle de battements et son frontal log-mel.

Sorties (dans le répertoire donné) :
- ``beat_this-final0.onnx`` : ``spect[1,T,128]`` -> logits ``beat[1,T]`` et ``downbeat[1,T]`` ;
- ``log-mel.onnx`` : onde mono 22 050 Hz ``wave[N]`` -> ``spect[T,128]`` (``torchaudio.MelSpectrogram`` puis
  ``log1p(1000·x)``, 50 trames par seconde), le frontal de ``beat_this.preprocessing.LogMelSpect``.

Usage : ``uv run export.py <checkpoint final0.ckpt> <répertoire de sortie> [<fichier audio de parité>]``.
Avec un fichier audio, l'écart entre torch et ONNX Runtime est mesuré (frontal puis modèle) et doit rester sous 1e-3 (frontal) et 1e-4 (modèle).
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from beat_this.inference import load_model
from beat_this.preprocessing import LogMelSpect

SAMPLE_RATE = 22_050
# Le frontal est en échelle log1p(1000·x) sur des valeurs jusqu'à ~10 : sa transformée de Fourier (opérateur DFT d'ONNX
# Runtime) diffère de celle de torch de quelques 1e-4 ; le modèle, lui, est comparé à 1e-4. Le critère qui compte est
# la grille obtenue de bout en bout, mesurée par le consommateur.
FRONTEND_TOLERANCE = 1e-3
MODEL_TOLERANCE = 1e-4


class Model(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, spect: torch.Tensor):
        prediction = self.model(spect)
        return prediction["beat"], prediction["downbeat"]


class Frontend(torch.nn.Module):
    """``LogMelSpect`` de beat_this tel quel : la parité est celle de l'exporteur, pas d'une réécriture."""

    def __init__(self):
        super().__init__()
        self.log_mel = LogMelSpect()

    def forward(self, wave: torch.Tensor):
        return self.log_mel(wave)


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def main() -> None:
    checkpoint, out = Path(sys.argv[1]), Path(sys.argv[2])
    audio = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)

    model = Model(load_model(str(checkpoint), "cpu")).eval()
    frontend = Frontend().eval()

    model_path = out / "beat_this-final0.onnx"
    torch.onnx.export(
        model,
        (torch.randn(1, 1500, 128),),
        str(model_path),
        input_names=["spect"],
        output_names=["beat", "downbeat"],
        dynamic_axes={"spect": {1: "time"}, "beat": {1: "time"}, "downbeat": {1: "time"}},
        opset_version=17,
        dynamo=False,
    )
    frontend_path = out / "log-mel.onnx"
    # Exporteur « dynamo » : `torch.stft` y devient l'opérateur DFT (FFT), que l'exporteur TorchScript ne sait pas
    # écrire avec une longueur d'onde dynamique.
    torch.onnx.export(
        frontend,
        (torch.randn(SAMPLE_RATE * 10),),
        dynamic_shapes={"wave": {0: torch.export.Dim("samples", min=SAMPLE_RATE, max=SAMPLE_RATE * 1300)}},
        input_names=["wave"],
        output_names=["spect"],
        opset_version=20,
        dynamo=True,
    ).save(str(frontend_path))

    if audio is None:
        return
    import subprocess

    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(audio), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        check=True,
        capture_output=True,
    ).stdout
    wave = np.frombuffer(pcm, dtype=np.float32).copy()
    with torch.inference_mode():
        # Référence : le `LogMelSpect` de beat_this lui-même, pas la réécriture exportée.
        reference = LogMelSpect()(torch.from_numpy(wave))
        reference_beat, reference_downbeat = model(reference[:1500].unsqueeze(0))
    spect = session(frontend_path).run(None, {"wave": wave})[0]
    frames = min(len(spect), len(reference))
    frontend_gap = float(np.abs(spect[:frames] - reference.numpy()[:frames]).max())
    beat, downbeat = session(model_path).run(None, {"spect": reference[:1500].unsqueeze(0).numpy()})
    model_gap = max(
        float(np.abs(beat - reference_beat.numpy()).max()),
        float(np.abs(downbeat - reference_downbeat.numpy()).max()),
    )
    print(f"frontal : {len(spect)} trames (torch {len(reference)}), écart max {frontend_gap:.2e}")
    print(f"modèle  : écart max des logits {model_gap:.2e}")
    if frontend_gap > FRONTEND_TOLERANCE or model_gap > MODEL_TOLERANCE or len(spect) != len(reference):
        raise SystemExit(f"parité hors tolérance (frontal {FRONTEND_TOLERANCE:g}, modèle {MODEL_TOLERANCE:g})")


if __name__ == "__main__":
    main()
