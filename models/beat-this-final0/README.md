# beat-this-final0

[beat_this](https://github.com/CPJKU/beat_this) `final0` (CPJKU, MIT) converti en ONNX : suivi des **battements** et des
**débuts de mesure** d'une musique. Utilisé par le converter de Jinto pour la grille rythmique de la bibliothèque musicale.

## Fichiers de la Release `beat-this-final0-v1`

| Fichier                  | Contenu                                                                          |
| ------------------------ | -------------------------------------------------------------------------------- |
| `log-mel.onnx`           | frontal : onde mono 22 050 Hz `wave[N]` → spectrogramme log-mel `spect[T,128]`      |
| `beat_this-final0.onnx`  | modèle : `spect[1,T,128]` → logits `beat[1,T]` et `downbeat[1,T]`                  |
| `beat_this-final0.ckpt`  | point de contrôle d'origine (PyTorch Lightning), pour rejouer la conversion     |
| `SHA256SUMS`             | empreintes SHA-256 de ces fichiers                                                  |

Le frontal est `beat_this.preprocessing.LogMelSpect` exporté tel quel par l'exporteur « dynamo » de PyTorch (opset 20,
`torch.stft` devient l'opérateur DFT) : transformée de Fourier à 1024 points (fenêtre de Hann, saut de 441 échantillons,
onde complétée par réflexion), module (`power=1`) divisé par √1024 (`normalized="frame_length"`), banc mel Slaney de 128
bandes entre 30 et 11 000 Hz, puis `log1p(1000·x)`. Le débit est de 50 trames par seconde ; le modèle a été entraîné sur
des morceaux de 1 500 trames. L'onde peut durer de 1 s à 20 min.

**Ce que fait le consommateur** (comme `split_predict_aggregate` et `Postprocessor("minimal")` de beat_this) :
découper le spectrogramme en morceaux de 1 500 trames avec un bord de 6 trames, garder les premières prédictions en cas
de recouvrement, puis retenir les maxima locaux sur ±3 trames dont le logit est positif, fusionner les pics adjacents et
ramener chaque début de mesure au battement le plus proche. Le temps d'une trame est de 20 ms.

## Parité mesurée

Écart maximal entre PyTorch et ONNX Runtime 1.30, sur trois musiques réelles de 2 à 3 minutes :

- frontal : de 7·10⁻⁵ à 5,9·10⁻⁴ sur le log-mel (valeurs jusqu'à 10 environ) : l'opérateur DFT d'ONNX Runtime n'a pas
  exactement l'arithmétique de la FFT de torch ; même nombre de trames des deux côtés ;
- modèle : de 3,9·10⁻⁵ à 8·10⁻⁵ sur les logits d'un morceau de 1 500 trames.

`export.py` échoue au-delà de 10⁻³ (frontal) et 10⁻⁴ (modèle). Le critère qui compte est la grille obtenue de bout en bout :
sur 13 musiques (dont trois à grille irrégulière), le consommateur de Jinto retrouve les battements et les débuts de
mesure de `analyze.py` à 20 ms près.

## Reproduire

```bash
uv sync
curl -LO https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt   # ou l'archive de la Release
sha256sum final0.ckpt   # 8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331
uv run export.py final0.ckpt out/ musique.mp3   # parité sur musique.mp3
uv run analyze.py final0.ckpt musique.mp3         # grille de référence (ms)
```
