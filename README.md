# Comparaison des algorithmes de compression

Outil en ligne de commande permettant de comparer les algorithmes de compression RCL, LZ77, Huffman et Arithmétique sur différents fichiers et images.

## Utilisation

```bash
python main.py <chemin> -a <algorithme> [options]
```

## Options disponibles

| Option | Description |
|---|---|
| `chemin` | Chemin vers un fichier ou un dossier à analyser. |
| `-a`, `--algorithm` | Algorithme à utiliser : `RCL`, `LZ77`, `HUFFMAN`, `ARITHMETIC` ou `ALL`. |
| `-o` | Dossier de destination des résultats CSV. |
| `--recursive` | Inclure les fichiers des sous-dossiers. |

## Exemples

**Exécuter tous les algorithmes sur un fichier :**

```bash
python main.py image.png -a ALL
```

**Exécuter tous les algorithmes sur un dossier :**

```bash
python main.py ./images -a ALL
```

**Exécuter un algorithme spécifique :**

```bash
python main.py ./images -a HUFFMAN
```

**Inclure les sous-dossiers :**

```bash
python main.py ./images -a ALL --recursive
```

**Spécifier le dossier de sortie :**

```bash
python main.py ./images -a ALL -o ./results
```

## Fichiers CSV générés

| Fichier | Description |
|---|---|
| `compression_results.csv` | Résultats détaillés pour chaque fichier et algorithme : tailles, taux de compression, temps d'exécution et entropies. |
| `comparison_by_file.csv` | Comparaison des taux de compression de chaque algorithme pour chaque fichier. |
| `comparison_by_algorithm.csv` | Statistiques globales permettant de comparer les performances des algorithmes. |

## Informations supplémentaires

- Les images sont converties en niveaux de gris sur 8 bits avant la compression.
- Les tailles compressées incluent les en-têtes nécessaires à la décompression.
- Les données décompressées sont comparées aux données originales afin de vérifier que la compression est sans perte.
- Un taux de compression (%) plus faible indique une meilleure compression.

Le taux de compression est calculé selon la formule :

\[
T = \frac{\text{Taille compressée}}{\text{Taille originale}}\times 100
\]

Par exemple, un taux de 40 % signifie que les données compressées occupent 40 % de leur taille originale, soit une réduction de 60 %.
