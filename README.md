# Compression d'images — APP3

Programme Python permettant de compresser et décompresser des fichiers à l'aide des algorithmes **RCL, LZ77, Huffman et arithmétique**.

## Prérequis

Installer les dépendances :

```bash
pip install numpy matplotlib pillow sortedcontainers graphviz
```

## 1. `main.py` — Analyse et comparaison des algorithmes

Permet de compresser un fichier ou un dossier complet, de comparer les performances des algorithmes et de générer des statistiques au format CSV.

**Exécuter tous les algorithmes sur un dossier :**

```bash
python main.py ./images -a ALL
```

**Exécuter un algorithme spécifique :**

```bash
python main.py image.png -a HUFFMAN
```

**Inclure les sous-dossiers et choisir le dossier de sortie :**

```bash
python main.py ./images -a ALL --recursive -o ./resultats
```

Options disponibles :

| Option | Description |
|---|---|
| `chemin` | Fichier ou dossier à analyser |
| `-a`, `--algorithm` | `RCL`, `LZ77`, `HUFFMAN`, `ARITHMETIC`, `ALL` |
| `-o` | Dossier de sortie des résultats |
| `--recursive` | Inclure les sous-dossiers |

Les CSV générés contiennent les tailles, les taux de compression, les entropies, les débits en bits/symbole (en-têtes inclus) et les temps d'exécution.

## 2. `main_images.py` — Compression de quatre images

Permet de regrouper quatre images compressées dans un seul fichier, puis de les reconstruire.

**Compresser quatre images :**

```bash
python main_images.py compresser image1.png image2.png image3.png image4.png -o images.comp -a AUTO
```

**Décompresser les quatre images :**

```bash
python main_images.py decompresser images.comp -o ./images_reconstruites -q 4
```

Options disponibles :

| Option | Description |
|---|---|
| `compresser` | Compresser quatre images dans un fichier |
| `decompresser` | Reconstruire les images depuis un fichier compressé |
| `-a` | `AUTO`, `RCL`, `LZ77`, `HUFFMAN`, `ARITHMETIC` |
| `-o` | Fichier de sortie (compression) ou dossier de sortie (décompression) |
| `-q` | Nombre d'images à reconstruire (1 à 4) |

Avec `AUTO`, le programme sélectionne automatiquement l'algorithme produisant le plus petit résultat pour chaque image.

## Remarques

- Les images sont converties en niveaux de gris sur 8 bits.
- Les fichiers compressés conservent leurs en-têtes nécessaires à la décompression.
- La compression est sans perte : les pixels reconstruits sont identiques aux pixels originaux après conversion en niveaux de gris.
- Un taux de compression plus faible indique une meilleure compression.
