import argparse
import csv
import struct
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from PIL import Image

import arithmetic
import huffman
import lz77
import rcl
import statistics  # Your local statistics.py containing SourceAnalyzer

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
ALGORITHMS = {
    'RCL': (rcl.compress, rcl.decompress),
    'LZ77': (lz77.compress, lz77.decompress),
    'ARITHMETIC': (arithmetic.compress, arithmetic.decompress),
    'HUFFMAN': (huffman.compress, huffman.decompress),
}


def load_source(path: Path) -> tuple[NDArray[np.uint8], bool, int, int]:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        with Image.open(path) as img:
            pixels = np.asarray(img.convert('L'), dtype=np.uint8)
            height, width = pixels.shape
            return pixels.ravel().copy(), True, width, height
    return np.frombuffer(path.read_bytes(), dtype=np.uint8).copy(), False, 0, 0


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def benchmark(path: Path, label: str, algorithms: list[str], plots: bool) -> list[dict]:
    data, is_image, width, height = load_source(path)
    if data.size == 0:
        print(f'Fichier vide ignoré : {label}')
        return []

    analyzer = statistics.SourceAnalyzer(data)
    if plots:
        analyzer.print_results()
        analyzer.plot_symbols()
        analyzer.plot_pairs()

    source_stats = {
        'fichier': label,
        'type_source': 'image (niveaux de gris)' if is_image else 'binaire',
        'taille_originale_octets': int(data.size),
        'symboles_distincts': int(np.count_nonzero(analyzer.symbol_counts)),
        'entropie_symbole_bits': analyzer.entropy_symbol(),
        'entropie_paire_bits': analyzer.entropy_pair() if data.size >= 2 else '',
        'entropie_conditionnelle_bits': analyzer.entropy_conditional() if data.size >= 2 else '',
    }

    results = []
    for name in algorithms:
        compress, decompress = ALGORITHMS[name]
        row = {**source_stats, 'algorithme': name}
        try:
            start = perf_counter()
            compressed = compress(data)
            compression_ms = (perf_counter() - start) * 1000

            # Les dimensions de l’image sont stockées dans un en-tête de 8 octets.
            # Inclure cet en-tête dans la taille compressée.
            if is_image:
                header = np.frombuffer(struct.pack('<II', width, height), dtype=np.uint8)
                packed = np.concatenate((header, compressed))
                compressed_payload = packed[8:]
            else:
                packed = compressed
                compressed_payload = packed

            start = perf_counter()
            restored = decompress(compressed_payload)
            decompression_ms = (perf_counter() - start) * 1000

            if not np.array_equal(data, restored):
                raise ValueError('Les données décompressées diffèrent des données originales')

            compressed_bytes = len(packed)
            ratio = 100 * compressed_bytes / data.size
            row.update({
                'taille_compressee_octets': compressed_bytes,
                'bits_par_symbole_comprime': round(8 * compressed_bytes / data.size, 4),
                'taux_compression_pct': round(ratio, 4),
                'gain_espace_pct': round(100 - ratio, 4),
                'temps_compression_ms': round(compression_ms, 4),
                'temps_decompression_ms': round(decompression_ms, 4),
                'verification': True,
                'erreur': '',
            })
            print(f'  {name:12} {ratio:8.2f}%  {compression_ms:10.2f} ms  OK')
        except Exception as exc:
            row.update({
                'taille_compressee_octets': '', 'bits_par_symbole_comprime': '', 'taux_compression_pct': '',
                'gain_espace_pct': '', 'temps_compression_ms': '',
                'temps_decompression_ms': '', 'verification': False,
                'erreur': f'{type(exc).__name__}: {exc}',
            })
            print(f'  {name:12} ÉCHEC : {exc}')
        results.append(row)
    return results


def export_results(rows: list[dict], output_dir: Path, algorithms: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_columns = [
        'fichier', 'type_source', 'algorithme', 'taille_originale_octets',
        'taille_compressee_octets', 'bits_par_symbole_comprime', 'taux_compression_pct', 'gain_espace_pct',
        'temps_compression_ms', 'temps_decompression_ms', 'verification', 'erreur',
        'symboles_distincts', 'entropie_symbole_bits', 'entropie_paire_bits',
        'entropie_conditionnelle_bits',
    ]
    write_csv(output_dir / 'resultats_compression.csv', rows, detail_columns)

    by_file = defaultdict(dict)
    for row in rows:
        by_file[row['fichier']][row['algorithme']] = row

    comparison_rows = []
    for filename, entries in sorted(by_file.items()):
        first = next(iter(entries.values()))
        item = {'fichier': filename, 'taille_originale_octets': first['taille_originale_octets'],
                'entropie_symbole_bits': first['entropie_symbole_bits'],
                'entropie_paire_bits': first['entropie_paire_bits'],
                'entropie_conditionnelle_bits': first['entropie_conditionnelle_bits']}
        for name in algorithms:
            result = entries.get(name, {})
            item[f'{name}_bits_par_symbole_comprime'] = result.get('bits_par_symbole_comprime', '')
            item[f'{name}_taux_compression_pct'] = result.get('taux_compression_pct', '')
            item[f'{name}_temps_compression_ms'] = result.get('temps_compression_ms', '')
            item[f'{name}_temps_decompression_ms'] = result.get('temps_decompression_ms', '')
        comparison_rows.append(item)

    comparison_columns = ['fichier', 'taille_originale_octets', 'entropie_symbole_bits', 'entropie_paire_bits', 'entropie_conditionnelle_bits']
    for name in algorithms:
        comparison_columns += [f'{name}_bits_par_symbole_comprime', f'{name}_taux_compression_pct', f'{name}_temps_compression_ms',
                               f'{name}_temps_decompression_ms']
    write_csv(output_dir / 'comparaison_par_fichier.csv', comparison_rows, comparison_columns)

    aggregate_rows = []
    for name in algorithms:
        valid = [r for r in rows if r['algorithme'] == name and r['verification']]
        failures = sum(r['algorithme'] == name and not r['verification'] for r in rows)
        total_original = sum(r['taille_originale_octets'] for r in valid)
        total_compressed = sum(r['taille_compressee_octets'] for r in valid)
        total_compress_ms = sum(r['temps_compression_ms'] for r in valid)
        total_decompress_ms = sum(r['temps_decompression_ms'] for r in valid)
        weighted_ratio = 100 * total_compressed / total_original if total_original else ''
        aggregate_rows.append({
            'algorithme': name,
            'fichiers_reussis': len(valid),
            'fichiers_echoues': failures,
            'total_original_octets': total_original,
            'total_compresse_octets': total_compressed,
            'taux_moyen_pct': round(float(np.mean([r['taux_compression_pct'] for r in valid])), 4) if valid else '',
            'taux_global_pct': round(weighted_ratio, 4) if valid else '',
            'bits_par_symbole_global': round(8 * total_compressed / total_original, 4) if total_original else '',
            'gain_global_pct': round(100 - weighted_ratio, 4) if valid else '',
            'temps_total_compression_ms': round(total_compress_ms, 4),
            'temps_total_decompression_ms': round(total_decompress_ms, 4),
            'debit_compression_Mo_s': round(total_original / (1000 * total_compress_ms), 4) if total_compress_ms else '',
            'debit_decompression_Mo_s': round(total_original / (1000 * total_decompress_ms), 4) if total_decompress_ms else '',
        })

    aggregate_columns = [
        'algorithme', 'fichiers_reussis', 'fichiers_echoues', 'total_original_octets',
        'total_compresse_octets', 'taux_moyen_pct', 'taux_global_pct',
        'bits_par_symbole_global',
        'gain_global_pct', 'temps_total_compression_ms', 'temps_total_decompression_ms',
        'debit_compression_Mo_s', 'debit_decompression_Mo_s',
    ]
    write_csv(output_dir / 'comparaison_par_algorithme.csv', aggregate_rows, aggregate_columns)


def parse_args():
    parser = argparse.ArgumentParser(description='Comparer les algorithmes de compression sur des fichiers ou des dossiers.')
    parser.add_argument('fichier', type=Path, help='Fichier ou dossier à analyser')
    parser.add_argument('-a', '--algorithm', type=str.upper,
                        choices=['ALL', *ALGORITHMS], default='ALL')
    parser.add_argument('-o', '--output', type=Path, default=Path('benchmark_results'),
                        help='Dossier de destination des résultats CSV')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Inclure les sous-dossiers')
    parser.add_argument('--plots', action='store_true',
                        help='Afficher les histogrammes de chaque source')
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.fichier.resolve()
    output_dir = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    if source.is_file():
        paths = [source]
        root = source.parent
    elif source.is_dir():
        paths = sorted(p for p in (source.rglob('*') if args.recursive else source.iterdir())
                       if p.is_file() and not p.resolve().is_relative_to(output_dir))
        root = source
    else:
        raise ValueError('La source doit être un fichier ou un dossier')

    algorithms = list(ALGORITHMS) if args.algorithm == 'ALL' else [args.algorithm]
    rows = []
    for path in paths:
        label = str(path.relative_to(root))
        print(f'\nFichier : {label}')
        try:
            rows.extend(benchmark(path, label, algorithms, args.plots))
        except Exception as exc:
            print(f'  Ignoré : {type(exc).__name__}: {exc}')

    if not rows:
        print('Aucun résultat à exporter.')
        return

    export_results(rows, output_dir, algorithms)
    print(f'\n{len(rows)} résultats exportés vers {output_dir}')
    print('  resultats_compression.csv')
    print('  comparaison_par_fichier.csv')
    print('  comparaison_par_algorithme.csv')


if __name__ == '__main__':
    main()
