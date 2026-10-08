# jinto-models

Modèles d'apprentissage automatique de Jinto, **convertis et publiés tels quels** pour être embarqués dans des images
(ONNX Runtime), sans Python à l'exécution. Le dépôt ne contient ni poids ni code de modèle amont : seulement le script
qui les convertit, ses dépendances épinglées, la documentation et la licence d'origine. Les fichiers eux-mêmes sont les
pièces jointes des **Releases**.

## Structure

```
models/<nom>/
  README.md        provenance, entrées et sorties, parité mesurée
  export.py        conversion (commit amont et empreinte du point de contrôle épinglés)
  pyproject.toml   dépendances de l'export ; uv.lock les fige
  uv.lock
  NOTICE           licence et provenance de l'amont (+ copie de sa licence)
```

## Releases

Une Release par modèle et par version, étiquetée `<nom>-v<N>` (ex. `beat-this-final0-v1`). Elle porte :

- les fichiers ONNX à embarquer ;
- le point de contrôle d'origine, en archive : si l'amont disparaît, la conversion reste rejouable ;
- `SHA256SUMS` : l'empreinte de chaque fichier. Les consommateurs tirent chaque fichier **par son empreinte**
  (`ADD --checksum=sha256:… <url>` dans un Dockerfile) ; une Release publiée ne change jamais, un changement de modèle ou
  de conversion est une nouvelle Release.

## Licences

Seuls des poids **redistribuables** entrent ici (MIT, Apache-2.0, BSD, CC-BY) : la licence du modèle est vérifiée avant
d'ajouter un dossier, et sa mention d'origine est reprise dans son `NOTICE`. Un modèle « non commercial » ou « research
only » n'a pas sa place ici.

## Ajouter un modèle

1. Créer `models/<nom>/` avec les fichiers ci-dessus ; épingler le commit amont et l'empreinte du point de contrôle.
2. Mesurer la parité entre le modèle d'origine et sa conversion sur un cas réel, et la consigner dans le README du modèle.
3. Publier la Release `<nom>-v1` avec `gh release create`, après avoir calculé `SHA256SUMS`.
