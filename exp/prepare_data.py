#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_data.py - unify dataset sources into one .pt contract:
  node_texts | x, edge_index, y, split(official, optional)

Sources:
  --source tgrb   : .pt from TGRB/LLMNodeBed (fields: raw_texts / x / edge_index / y ...)
  --source taglas : dataset key from TAGLAS (cora_node / pubmed_node / wikics / arxiv / ...)

The TAGLAS path is a one-time wiring job: TAGLAS structures its data through
global text pools + *_map indices, and exact attribute layout depends on version.
This script tries the most common layouts and prints exactly what it found, so a
failed run is a 2-minute fix instead of a black box. Copy the error output back
if you hit a layout we did not anticipate.
"""
import argparse, os, torch


def save(d, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(d, out)
    print(f"saved unified data -> {out}")


def from_tgrb(path):
    g = torch.load(path, weights_only=False)
    keys = list(g.keys()) if hasattr(g, "keys") else dir(g)
    print("[tgrb] keys found:", keys)
    edge_index = g.get("edge_index") or g.edge_index
    y = g.get("y") if g.get("y") is not None else g.y
    texts = g.get("raw_texts") if isinstance(g.get("raw_texts"), list) else None
    if texts is None and hasattr(g, "raw_texts") and isinstance(g.raw_texts, list):
        texts = g.raw_texts
    x = torch.as_tensor(g.get("x"), dtype=torch.float32) if g.get("x") is not None else \
        (torch.as_tensor(g.x, dtype=torch.float32) if hasattr(g, "x") and g.x is not None else None)
    if texts is None and x is None:
        raise ValueError("need raw_texts(list) or x(tensor) in TGRB .pt")
    split = {}
    for k in ("train_mask", "val_mask", "test_mask"):
        if g.get(k) is not None or hasattr(g, k):
            split[k.replace("_mask", "")] = torch.as_tensor(g.get(k) if g.get(k) is not None else getattr(g, k), dtype=torch.bool)
    save({"node_texts": texts, "x": x, "edge_index": torch.as_tensor(edge_index, dtype=torch.long),
          "y": torch.as_tensor(y, dtype=torch.long), "split": split or None}, args.out)


def from_taglas(ds):
    try:
        from TAGLAS import get_dataset
    except ImportError:
        raise SystemExit("pip install git+https://github.com/JiaruiFeng/TAGLAS  first, "
                         "and run its own data-download trigger once (first get_dataset call).")
    dataset = get_dataset(ds, root="./TAGDataset")
    n = dataset.num_nodes if hasattr(dataset, "num_nodes") else None
    print("[taglas] class:", type(dataset).__name__, "num_nodes:", n)
    # common layout: global text pools + map keys on each sample Data
    sample = dataset[0] if hasattr(dataset, "__getitem__") and not hasattr(dataset, "data") else getattr(dataset, "data", None)
    print("[taglas] sample keys:", list(sample.keys) if hasattr(sample, "keys") else "n/a")
    # Node-level graph data (node classification) usually exposes:
    #   x (global text pool), node_map (per-graph node -> pool idx), edge_index, label_map/y
    raise SystemExit(
        "TAGLAS wiring TODO (one-time): inspect the printed sample keys above, then assemble:\n"
        "  node_texts = [pool_x[i] for i in node_map]  (pool_x = dataset.x text list)\n"
        "  y          = labels via label_map; edge_index from sample\n"
        "and call save({...}, out). Paste the 'sample keys' output to the assistant to finish the adapter."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["tgrb", "taglas"])
    ap.add_argument("--path", default=None, help="tgrb: path to .pt")
    ap.add_argument("--dataset", default=None, help="taglas: dataset key e.g. cora_node")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.source == "tgrb":
        from_tgrb(args.path)
    else:
        from_taglas(args.dataset)
