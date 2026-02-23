"""Visualize landmark overlays and simple feature-importance estimates."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.feature_extraction import extract_geometric_analysis_from_image


def draw_landmarks(image_path, out_path):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    analysis = extract_geometric_analysis_from_image(img)
    lm = analysis['landmarks']

    plt.figure(figsize=(8, 8))
    plt.imshow(arr)
    for name, xy in lm.items():
        plt.scatter([xy[0]], [xy[1]], s=20)
        plt.text(xy[0] + 2, xy[1] + 2, name, fontsize=7)

    def line(a, b, c):
        if a in lm and b in lm:
            xa, ya = lm[a]
            xb, yb = lm[b]
            plt.plot([xa, xb], [ya, yb], c=c, linewidth=2)

    line('left_eye', 'right_eye', 'lime')
    line('nose_tip', 'mouth_center', 'cyan')
    line('left_jaw', 'right_jaw', 'red')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def feature_importance_placeholder(distances, out_path):
    keys = list(distances.keys())
    vals = np.abs(np.array(list(distances.values()), dtype=np.float32))
    plt.figure(figsize=(10, 4))
    plt.bar(keys, vals)
    plt.xticks(rotation=45, ha='right')
    plt.title('Geometric Feature Magnitudes (proxy importance)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--overlay_out', default='landmark_overlay.png')
    ap.add_argument('--importance_out', default='feature_importance.png')
    args = ap.parse_args()

    img = Image.open(args.image).convert('RGB')
    analysis = extract_geometric_analysis_from_image(img)
    draw_landmarks(args.image, args.overlay_out)
    feature_importance_placeholder(analysis['distances'], args.importance_out)
