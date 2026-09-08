#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
taglas_bench.py - embed->GNN robustness baseline harness (scaffold).

Data contract (unified .pt produced by prepare_data.py):
  node_texts : List[str] | None
  x          : Tensor[N,d] | None
  edge_index : LongTensor[2,E]
  y          : LongTensor[N]
  split      : dict | None  ('official' masks)

Usage examples (see exp/README.md):
  python taglas_bench.py --dataset cora_node --encoder minilm --model gcn --attack none --seeds 3
  python taglas_bench.py --dataset cora_node --encoder minilm --model cosgcn --attack pgd_evasion --ptb 0.2
  python taglas_bench.py --dataset cora_node --encoder roberta --model gcn --attack text_evasion --ptb 0.4

NOTE: scaffold not yet GPU-verified; run S0 smoke first.
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ENCODER_NAMES = {"minilm": "all-MiniLM-L6-v2", "roberta": "all-roberta-large-v1"}


# ---------------------------------------------------------------- models
class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.drop = torch.nn.Dropout(dropout)
        self.lin1 = torch.nn.Linear(in_dim, hidden)
        self.lin2 = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index=None):
        x = self.drop(torch.relu(self.lin1(x)))
        return self.lin2(x)


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.drop = torch.nn.Dropout(dropout)
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def forward(self, x, edge_index):
        x = self.drop(torch.relu(self.conv1(x, edge_index)))
        return self.conv2(x, edge_index)


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        from torch_geometric.nn import GATConv
        h = hidden // 8
        self.drop = torch.nn.Dropout(dropout)
        self.conv1 = GATConv(in_dim, h, heads=8)
        self.conv2 = GATConv(h * 8, num_classes, heads=1)

    def forward(self, x, edge_index):
        x = self.drop(F.elu(self.conv1(x, edge_index)))
        return self.conv2(x, edge_index)


class GPRGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout, alpha=0.1, K=10):
        super().__init__()
        from torch_geometric.nn import GPRConv
        self.lin1 = torch.nn.Linear(in_dim, hidden)
        self.lin2 = torch.nn.Linear(hidden, num_classes)
        self.drop = torch.nn.Dropout(dropout)
        self.conv = GPRConv(K, alpha, Init="PPR", cached=False)

    def forward(self, x, edge_index):
        x = self.drop(F.relu(self.lin1(x)))
        x = self.lin2(x)
        return self.conv(x, edge_index)


class CosGCN(torch.nn.Module):
    """GNNGuard-style static cosine edge gating (similarity-filter family).
    A lower bound / contrast for NSPGNN: expect robust to structure attack,
    fragile to text attack (features of attacked nodes shift -> gate breaks)."""

    def __init__(self, in_dim, hidden, num_classes, dropout, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.base = GCN(in_dim, hidden, num_classes, dropout)

    def gated_index(self, x, edge_index):
        src, dst = edge_index
        xn = F.normalize(x, dim=1)
        sim = (xn[src] * xn[dst]).sum(1)
        keep = sim >= self.threshold
        return edge_index[:, keep]

    def forward(self, x, edge_index):
        g = self.gated_index(x, edge_index)
        return self.base(x, g)


class NSPGNN(torch.nn.Module):
    """Placeholder: wire the group's NSPGNN source here.
    Expected interface: forward(x, edge_index) -> logits."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "NSPGNN not wired yet. Put your NSPGNN implementation in this class "
            "(forward(x, edge_index)->logits) and re-run. Mind O(N^2) dual-kNN cost "
            "on large graphs - use Lanczos/blocked construction or subsample."
        )


MODELS = {"mlp": MLP, "gcn": GCN, "gat": GAT, "gprgnn": GPRGNN, "cosgcn": CosGCN, "nspgnn": NSPGNN}


# ---------------------------------------------------------------- data / encode
def load_data(path):
    d = torch.load(path, weights_only=False)
    assert "edge_index" in d and "y" in d, f"bad contract in {path}"
    if d.get("node_texts") is None and d.get("x") is None:
        raise ValueError("need node_texts or x")
    if d.get("split") is not None:
        for k, v in d["split"].items():
            d["split"][k] = torch.as_tensor(v, dtype=torch.bool)
    return d


def make_split(n, y, kind, seed):
    if kind == "official":
        return None
    rng = random.Random(seed)
    perm = torch.randperm(n)
    if kind == "inductive":  # stratified-ish 60/20/20 by perm (stratified per class)
        idx = {c: perm[y[perm] == c].tolist() for c in torch.unique(y).tolist()}
        train, val, test = [], [], []
        for c, lst in idx.items():
            rng.shuffle(lst)
            a = int(len(lst) * 0.6)
            b = int(len(lst) * 0.8)
            train += lst[:a]
            val += lst[a:b]
            test += lst[b:]
    else:  # transductive 10/10/80
        lst = perm.tolist()
        a, b = int(n * 0.1), int(n * 0.2)
        train, val, test = lst[:a], lst[a:b], lst[b:]
    mask = lambda l: torch.zeros(n, dtype=torch.bool).scatter_(0, torch.tensor(l), True)
    return {"train": mask(train), "val": mask(val), "test": mask(test)}


def get_features(d, encoder, cache_path, device):
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False).to(device)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if encoder == "bow":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=10000, stop_words="english")
        X = vec.fit_transform(d["node_texts"]).toarray()
        feat = torch.tensor(X, dtype=torch.float32)
    else:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        model = SentenceTransformer(ENCODER_NAMES[encoder], device="cpu")
        emb = model.encode(d["node_texts"], batch_size=256, show_progress_bar=True)
        feat = torch.tensor(np.asarray(emb), dtype=torch.float32)
        print(f"[encode] {encoder}: {len(d['node_texts'])} texts in {time.time()-t0:.0f}s")
    torch.save(feat.cpu(), cache_path)
    return feat.to(device)


# ---------------------------------------------------------------- lightweight attacks
def heuristic_perturb(edge_index, y, n, ptb, seed, mode="poison", device="cpu"):
    """DICE-ish heuristic: add edges between dissimilar (diff-class) pairs,
    remove edges between same-class pairs. Gradient-free, for smoke only."""
    rng = np.random.RandomState(seed)
    n_add = n_del = int(edge_index.shape[1] * ptb / 2)
    E = set(map(tuple, edge_index.t().tolist()))
    src, dst = edge_index
    same = (y[src] == y[dst]).numpy()
    cand_remove = edge_index.t()[np.where(same)[0]].tolist()
    add = []
    yv = y.tolist()
    while len(add) < n_add:
        u, v = int(rng.randint(0, n)), int(rng.randint(0, n))
        if u != v and yv[u] != yv[v] and (u, v) not in E and (v, u) not in E:
            E.add((u, v))
            add.append((u, v))
    rng.shuffle(cand_remove)
    removed = cand_remove[:n_del]
    for u, v in removed:
        E.discard((u, v))
    all_e = list(E)  # E already holds original (minus removed) + added edges
    ei = torch.tensor(all_e, dtype=torch.long).t().contiguous()
    return ei


def add_low_sim_edges(edge_index, x, n, n_add, seed, device):
    """Add edges between lowest cosine-similarity node pairs (attack preference
    per NSPGNN Theorem 1 style). Smoke proxy of PGD."""
    rng = np.random.RandomState(seed)
    xn = F.normalize(x, dim=1)
    sim = xn @ xn.t()
    sim.fill_diagonal_(-1)
    if n > 20000:  # large graph: subsample candidates
        cand = torch.randint(0, n, (min(n, 20000),), device=device)
        sim = sim[cand]
        base = cand
    else:
        base = torch.arange(n, device=device)
    flat = sim.flatten()
    k = min(n_add * 3, flat.numel())
    top = torch.topk(flat, k, largest=False).indices
    E = set(map(tuple, edge_index.t().tolist()))
    out, got = [], 0
    for idx in top.tolist():
        u = int(base[idx // sim.shape[1]])
        v = int(idx % sim.shape[1])
        if u != v and (u, v) not in E and (v, u) not in E:
            E.add((u, v))
            out.append((u, v))
            got += 1
            if got >= n_add:
                break
    new_e = list(map(tuple, edge_index.t().tolist())) + out
    return torch.tensor(new_e, dtype=torch.long).t().contiguous()


# ---------------------------------------------------------------- text attack (local vLLM)
def llm_rewrite_attack(d, masks, ptb, split_kind, cfg, attacked_out):
    """Neighborhood-aware rewrite (TGRB style) via local vLLM OpenAI endpoint."""
    import openai
    client = openai.OpenAI(base_url=cfg["vllm"]["url"], api_key="EMPTY")
    pool = masks["train"] if split_kind == "transductive" else masks["test"]
    node_ids = pool.nonzero().flatten().tolist()
    n_atk = max(1, int(len(node_ids) * ptb))
    rng = random.Random(0)
    degrees = torch.bincount(d["edge_index"][0], minlength=len(d["y"]))
    w = (1.0 / (degrees[node_ids].float() + 1.0)).tolist() if cfg["text_attack"]["low_degree_bias"] else None
    target = rng.choices(node_ids, weights=w, k=n_atk)
    idx2lab = {}
    yl = d["y"].tolist()
    for i in node_ids:
        idx2lab[i] = yl[i]
    counts = torch.bincount(d["y"])
    texts = list(d["node_texts"])
    import collections
    nb_cache = {}
    for vid in target:
        nb = d["edge_index"][1][d["edge_index"][0] == vid].tolist()
        nb = [u for u in nb if u in idx2lab][:cfg["text_attack"]["max_neighbors_in_prompt"]]
        nb_labels = collections.Counter(idx2lab[u] for u in nb)
        lab_counts = ", ".join(f"class {c}:{k}" for c, k in sorted(nb_labels.items()))
        txt = texts[vid][: cfg["text_attack"]["max_text_len"]]
        prompt = (
            "You are attacking a node classifier on a citation graph.\n"
            f"Node text:\n{txt}\nNode class: {idx2lab[vid]}\n"
            f"Neighbor class counts: {lab_counts}\n"
            "Rewrite the node text so a classifier misclassifies it: keep it fluent and similar length, "
            "but shift the topic away from its own class and from the dominant neighbor classes.\n"
            "Output only the rewritten text."
        )
        resp = client.chat.completions.create(
            model=cfg["vllm"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["vllm"]["temperature"], max_tokens=cfg["text_attack"]["max_text_len"],
        )
        new_txt = resp.choices[0].message.content.strip()
        if new_txt:
            texts[vid] = new_txt
    json.dump({str(i): texts[i] for i in target}, open(attacked_out, "w"))
    return texts, target


# ---------------------------------------------------------------- train / eval
def train_eval(model, x, edge_index, y, mask, device, cfg, epochs_scale=1.0):
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    best = (0.0, None)
    t = cfg["train"]
    patience = 0
    epochs = max(1, int(t["epochs"] * epochs_scale))
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[mask["train"]], y[mask["train"]])
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                acc = (out[mask["val"]].argmax(1) == y[mask["val"]]).float().mean().item()
            if acc > best[0]:
                best = (acc, {k: v.clone() for k, v in model.state_dict().items()})
                patience = 0
            else:
                patience += 10
                if patience >= t["patience"]:
                    break
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        test_acc = (out[mask["test"]].argmax(1) == y[mask["test"]]).float().mean().item()
    return test_acc


def run_one(args, cfg, d, seed):
    device = torch.device("cuda" if (cfg["device"] == "auto" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    n = d["y"].shape[0]
    mask = d.get("split") if args.split == "official" and d.get("split") else make_split(n, d["y"], args.split, seed)
    ei = d["edge_index"]
    feat = get_features(d, args.encoder, os.path.join(cfg["cache_dir"], args.dataset, args.encoder + ".pt"), device)
    attacked_texts = None
    if args.attack == "text_poison" and args.split == "transductive":
        cache = os.path.join(cfg["cache_dir"], args.dataset, f"text_atk_{args.ptb}.pt")
        if os.path.exists(cache):
            attacked_texts = d["node_texts"]
        else:
            print("[text_attack] calling local LLM (poisoning budget on train)...")
            texts, _ = llm_rewrite_attack(d, mask, args.ptb, args.split, cfg, "/tmp/atk.json")
            d2 = dict(d)
            d2["node_texts"] = texts
            feat = get_features(d2, args.encoder, cache, device)
            attacked_texts = texts
    if args.attack in ("pgd_evasion", "heur_poison"):
        ei = add_low_sim_edges(ei, feat, n, int(ei.shape[1] * args.ptb), seed, device) if args.attack == "pgd_evasion" \
            else heuristic_perturb(ei, d["y"], n, args.ptb, seed)
    cls = MODELS[args.model]
    if args.model == "cosgcn":
        model = cls(feat.shape[1], cfg["train"]["hidden"], int(d["y"].max()) + 1, cfg["train"]["dropout"], cfg["cosgcn"]["threshold"])
    elif args.model == "gprgnn":
        model = cls(feat.shape[1], cfg["train"]["hidden"], int(d["y"].max()) + 1, cfg["train"]["dropout"], cfg["train"]["gpr_alpha"])
    else:
        model = cls(feat.shape[1], cfg["train"]["hidden"], int(d["y"].max()) + 1, cfg["train"]["dropout"])
    acc = train_eval(model, feat, ei, d["y"], mask, device, cfg)
    return {"seed": seed, "acc": acc, "n_edges_atk": ei.shape[1], "text_attacked": attacked_texts is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--attack", default=None)
    ap.add_argument("--ptb", type=float, default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--vllm-url", default=None)
    ap.add_argument("--vllm-model", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for k in ["dataset", "encoder", "model", "attack", "split", "seeds"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.ptb is not None:
        cfg["ptb"] = args.ptb
    if args.vllm_url:
        cfg["vllm"]["url"] = args.vllm_url
    if args.vllm_model:
        cfg["vllm"]["model"] = args.vllm_model
    d = load_data(os.path.join(cfg["data_dir"], f"{cfg['dataset']}.pt"))
    results = [run_one(args, cfg, d, s) for s in range(cfg["seeds"])]
    accs = [r["acc"] for r in results]
    row = {"dataset": cfg["dataset"], "encoder": cfg["encoder"], "model": cfg["model"],
           "attack": cfg["attack"], "ptb": cfg["ptb"], "split": cfg["split"],
           "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)), "seeds": results}
    out_dir = os.path.join(cfg["out_dir"], cfg["dataset"], cfg["encoder"])
    os.makedirs(out_dir, exist_ok=True)
    name = f"{cfg['model']}_{cfg['attack']}_{cfg['ptb']}_{cfg['split']}.json"
    json.dump(row, open(os.path.join(out_dir, name), "w"), indent=2)
    print(f"| {cfg['dataset']} | {cfg['encoder']} | {cfg['model']} | {cfg['attack']} {cfg['ptb']} | "
          f"{cfg['split']} | {np.mean(accs):.2f}±{np.std(accs):.2f} | -> {os.path.join(out_dir, name)}")


if __name__ == "__main__":
    main()
