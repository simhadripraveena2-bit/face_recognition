"""Evaluation entrypoint for FERET/LFW style datasets.
Dataset structure expected: <root>/<person>/<image files>
"""

import argparse
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from src.inference import image_to_embedding, image_to_geometric_features


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_features(root):
    by_id = defaultdict(list)
    for person in os.listdir(root):
        pdir = os.path.join(root, person)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            path = os.path.join(pdir, fn)
            if not os.path.isfile(path):
                continue
            img = Image.open(path).convert('RGB')
            emb = image_to_embedding(img)
            if emb is None:
                continue
            geo = image_to_geometric_features(img)
            by_id[person].append((emb, geo))
    return by_id


def simple_rank1(by_id):
    gallery, labels = [], []
    for person, feats in by_id.items():
        if not feats:
            continue
        gallery.append(np.concatenate([feats[0][0], feats[0][1]]))
        labels.append(person)
    gallery = np.array(gallery)

    correct, total = 0, 0
    for person, feats in by_id.items():
        for emb, geo in feats[1:]:
            probe = np.concatenate([emb, geo])
            sims = [cosine(probe, g) for g in gallery]
            pred = labels[int(np.argmax(sims))]
            correct += int(pred == person)
            total += 1
    return correct / max(total, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw_root", type=str, default=None)
    ap.add_argument("--feret_root", type=str, default=None)
    args = ap.parse_args()

    if args.lfw_root:
        lfw = load_features(args.lfw_root)
        print("LFW rank-1:", simple_rank1(lfw))
    if args.feret_root:
        feret = load_features(args.feret_root)
        print("FERET rank-1:", simple_rank1(feret))
