# This main was generated in good part by chat GPT
"""Compression de quatre images dans un fichier commun selon le schéma APP3.

Format (big endian):
  4 x uint8 : identifiant d'algorithme
  4 x (uint32 largeur, uint32 hauteur) : dimensions (8 octets/image)
  4 x uint64 : longueur des blocs compressés (8 octets/image)
  4 blocs de données compressées, concaténés

Les codecs Huffman et arithmétique existants intègrent leur modèle au bloc.
Voir README pour la divergence avec les consignes de l'enseignant.
"""
import argparse
import struct
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

import arithmetic
import huffman
import lz77
import rcl
from statistics import SourceAnalyzer

ALGORITHMES = {
    'RCL': (0, rcl.compress, rcl.decompress),
    'HUFFMAN': (1, huffman.compress, huffman.decompress),
    'ARITHMETIC': (2, arithmetic.compress, arithmetic.decompress),
}
PAR_ID = {spec[0]: (name, spec[2]) for name, spec in ALGORITHMES.items()}
N_IMAGES = 4
ENTETE = struct.Struct('>4B' + 'II' * N_IMAGES + 'Q' * N_IMAGES)


def lire_image(chemin: Path):
    with Image.open(chemin) as image:
        image = image.convert('L')
        largeur, hauteur = image.size
        return np.asarray(image, dtype=np.uint8).ravel().copy(), largeur, hauteur


def compresser(chemins: list[Path], destination: Path, algorithme: str):
    if len(chemins) != N_IMAGES:
        raise ValueError('Il faut exactement quatre images.')
    identifiants, dimensions, longueurs, blocs = [], [], [], []
    for numero, chemin in enumerate(chemins, 1):
        pixels, largeur, hauteur = lire_image(chemin)
        analyse = SourceAnalyzer(pixels)
        candidats = [algorithme] if algorithme != 'AUTO' else list(ALGORITHMES)
        resultats = []
        for nom in candidats:
            identifiant, encoder, decoder = ALGORITHMES[nom]
            debut = perf_counter()
            code = encoder(pixels)
            duree = (perf_counter() - debut) * 1000
            # Valider avant d'enregistrer le bloc.
            if not np.array_equal(decoder(code), pixels):
                raise ValueError(f'Compression non réversible : {chemin}, {nom}')
            resultats.append((len(code), nom, identifiant, code, duree))
        taille, nom, identifiant, code, duree = min(resultats, key=lambda r: r[0])
        identifiants.append(identifiant)
        dimensions.extend([largeur, hauteur])
        longueurs.append(taille)
        blocs.append(code.tobytes())
        print(f'Image {numero} : {chemin.name} ({largeur} x {hauteur})')
        print(f'  Entropie H(X) : {analyse.entropy_symbol():.4f} bits/symbole')
        print(f'  Entropie H(X,Y) : {analyse.entropy_pair():.4f} bits/paire')
        print(f'  Compression : {nom} | {taille} octets | {duree:.2f} ms')
        print(f'  Débit du bloc : {8 * taille / pixels.size:.4f} bits/symbole')
        print(f'  Redondance par rapport à 8 bits : {100 * (1 - taille / pixels.size):.2f} %')
        if algorithme == 'AUTO':
            print('  Comparaison : ' + ', '.join(f'{r[1]}={8*r[0]/pixels.size:.4f} bits/symbole ({r[0]} octets)' for r in resultats))
    # Une seule écriture du fichier, après construction complète de l'en-tête et du corps.
    contenu = ENTETE.pack(*identifiants, *dimensions, *longueurs) + b''.join(blocs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(contenu)
    pixels_total = sum(dimensions[2*i] * dimensions[2*i+1] for i in range(N_IMAGES))
    print(f'\nFichier créé : {destination} ({len(contenu)} octets, en-tête {ENTETE.size} octets)')
    print(f'Débit total, en-tête compris : {8 * len(contenu) / pixels_total:.4f} bits/symbole')


def decompresser(source: Path, destination: Path, qualite: int):
    contenu = source.read_bytes()
    if len(contenu) < ENTETE.size:
        raise ValueError('Fichier trop court pour contenir un en-tête valide.')
    valeurs = ENTETE.unpack_from(contenu)
    ids = valeurs[:N_IMAGES]
    dimensions = valeurs[N_IMAGES:N_IMAGES + 2*N_IMAGES]
    longueurs = valeurs[N_IMAGES + 2*N_IMAGES:]
    if sum(longueurs) != len(contenu) - ENTETE.size:
        raise ValueError('Longueurs invalides ou fichier tronqué.')
    destination.mkdir(parents=True, exist_ok=True)
    position = ENTETE.size
    for i in range(N_IMAGES):
        taille = longueurs[i]
        bloc = contenu[position:position + taille]
        position += taille
        if i >= qualite:
            continue
        largeur, hauteur = dimensions[2*i:2*i+2]
        if largeur == 0 or hauteur == 0:
            raise ValueError('Dimensions invalides.')
        if ids[i] not in PAR_ID:
            raise ValueError(f'Algorithme inconnu : {ids[i]}')
        nom, decoder = PAR_ID[ids[i]]
        pixels = decoder(np.frombuffer(bloc, dtype=np.uint8))
        if pixels.size != largeur * hauteur:
            raise ValueError(f'Nombre de pixels incorrect pour image {i+1}')
        chemin = destination / f'image_{i+1}.png'
        Image.fromarray(pixels.reshape(hauteur, largeur)).save(chemin)
        print(f'Image {i+1} : {chemin} ({nom})')


def main():
    parser = argparse.ArgumentParser(description='Compresser quatre images dans un seul fichier ou les décompresser.')
    sous = parser.add_subparsers(dest='commande', required=True)
    comp = sous.add_parser('compresser', help='Créer un fichier contenant quatre images')
    comp.add_argument('images', type=Path, nargs=4, help='Quatre images dans l’ordre de qualité')
    comp.add_argument('-o', '--output', type=Path, required=True, help='Fichier compressé à créer')
    comp.add_argument('-a', '--algorithm', type=str.upper, choices=['AUTO', *ALGORITHMES], default='AUTO')
    dec = sous.add_parser('decompresser', help='Extraire les images')
    dec.add_argument('fichier', type=Path)
    dec.add_argument('-o', '--output', type=Path, default=Path('images_decompressees'))
    dec.add_argument('-q', '--qualite', type=int, choices=range(1, N_IMAGES+1), default=N_IMAGES,
                     help='Nombre d’images à extraire (1 à 4); vérifier la correspondance avec le guide étudiant')
    args = parser.parse_args()
    if args.commande == 'compresser':
        compresser(args.images, args.output, args.algorithm)
    else:
        decompresser(args.fichier, args.output, args.qualite)


if __name__ == '__main__':
    main()
