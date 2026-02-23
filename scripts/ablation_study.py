"""Ablation: embedding-only vs geometry-only vs hybrid."""

import argparse
import os
import numpy as np


def load_class_prototypes(ckpt_path):
    import torch

    ckpt = torch.load(ckpt_path, map_location='cpu')
    return np.asarray(ckpt.get('embedding_prototypes')), np.asarray(ckpt.get('geometry_prototypes'))


def summary(emb_proto, geo_proto):
    print("Embedding-only prototype shape:", emb_proto.shape)
    print("Geometry-only prototype shape:", geo_proto.shape)
    if emb_proto.ndim < 2 or geo_proto.ndim < 2:
        print("Checkpoint does not include prototype matrices yet. Re-train with current pipeline for full ablation outputs.")
        return
    print("Hybrid feature dim:", emb_proto.shape[1] + geo_proto.shape[1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/classifier_checkpoint.pth")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    emb, geo = load_class_prototypes(args.checkpoint)
    summary(emb, geo)
