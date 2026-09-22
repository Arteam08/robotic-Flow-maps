"""
FID pilot evaluation. Inception features via torch-fidelity's TF-compatible InceptionV3 (same net as ADM suite).
Reports per sample folder: FID_N, FID_inf (1/N extrapolation, Chong & Forsyth 2020), KID (unbiased MMD^2, poly kernel),
IS, top-1 accuracy of a frozen ResNet-50 on (x, y), with seed-bootstrap CIs.
"""
import argparse, os, json, glob, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

dev = "cuda"


def get_extractor():
    fe = FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"]).to(dev).eval()
    return fe


@torch.no_grad()
def feats_from_uint8(fe, arr_iter):
    f2048, logits = [], []
    for x in arr_iter:  # x: uint8 numpy NHWC
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).to(dev)
        a, b = fe(x)
        f2048.append(a.cpu()); logits.append(b.cpu())
    return torch.cat(f2048).numpy().astype(np.float64), torch.cat(logits).numpy()


def npz_batches(d, bs=250):
    arr = d["arr_0"]
    print("ref batch images", arr.shape, arr.dtype, flush=True)
    for i in range(0, len(arr), bs):
        yield arr[i:i + bs]


def folder_batches(folder, idx, bs=250):
    files = [f"{folder}/{i:06d}.png" for i in idx]
    for i in range(0, len(files), bs):
        yield np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files[i:i + bs]])


def available_indices(folder):
    return sorted(int(os.path.basename(f)[:6]) for f in glob.glob(f"{folder}/[0-9]*.png"))


def load_labels(folder, n):
    """{folder}_labels.npy (DDP run) or merge of {folder}_labels_shard*.npy [(index, y), ...]."""
    if os.path.exists(f"{folder}_labels.npy"):
        return np.load(f"{folder}_labels.npy")[:n]
    shards = sorted(glob.glob(f"{folder}_labels_shard*.npy"))
    if not shards:
        return None
    out = np.full(n, -1, dtype=np.int64)
    for s in shards:
        a = np.load(s)
        a = a[a[:, 0] < n]
        out[a[:, 0]] = a[:, 1]
    return out


def n_available(folder):
    k = 0
    while os.path.exists(f"{folder}/{k:06d}.png"):
        k += 1
    return k


class RefStats:
    def __init__(self, feats, mu=None, sigma=None):
        self.feats = feats
        self.mu = feats.mean(0) if mu is None else mu.astype(np.float64)
        self.sigma = np.cov(feats, rowvar=False) if sigma is None else sigma.astype(np.float64)
        w, V = np.linalg.eigh(self.sigma)
        self.sqrt_sigma = (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def fid_from_feats(ref: RefStats, feats):
    mu, sigma = feats.mean(0), np.cov(feats, rowvar=False)
    A = ref.sqrt_sigma @ sigma @ ref.sqrt_sigma
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh((A + A.T) / 2), 0, None)).sum()
    return float(((mu - ref.mu) ** 2).sum() + np.trace(sigma) + np.trace(ref.sigma) - 2 * tr_sqrt)


def fid_inf(ref, feats, rng, n_points=8, reps=3):
    N = len(feats)
    ns = np.unique(np.linspace(max(512, N // 4), N, n_points).astype(int))
    xs, ys = [], []
    for n in ns:
        for _ in range(reps if n < N else 1):
            sub = feats[rng.choice(N, n, replace=False)]
            xs.append(1.0 / n); ys.append(fid_from_feats(ref, sub))
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(intercept), float(slope), list(zip(map(float, xs), map(float, ys)))


def kid(ref_feats, feats, rng, subsets=100, subset_size=1000):
    d = feats.shape[1]
    m = min(subset_size, len(feats), len(ref_feats))
    vals = []
    for _ in range(subsets):
        x = feats[rng.choice(len(feats), m, replace=False)]
        y = ref_feats[rng.choice(len(ref_feats), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3; kyy = (y @ y.T / d + 1) ** 3; kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) + (kyy.sum() - np.trace(kyy))) / (m * (m - 1)) - 2 * kxy.mean()
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def inception_score(logits, splits=1):
    p = torch.softmax(torch.from_numpy(logits).double(), 1).numpy()
    scores = []
    for part in np.array_split(p, splits):
        py = part.mean(0, keepdims=True)
        kl = (part * (np.log(part + 1e-12) - np.log(py + 1e-12))).sum(1).mean()
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


@torch.no_grad()
def classifier_acc(folder, idx, labels):
    n = len(idx)
    import torchvision
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    net = torchvision.models.resnet50(weights=w).to(dev).eval()
    tf = w.transforms()
    correct, top5 = 0, 0
    for i in range(0, n, 128):
        imgs = [tf(Image.open(f"{folder}/{k:06d}.png").convert("RGB")) for k in idx[i:i + 128]]
        out = net(torch.stack(imgs).to(dev))
        y = torch.from_numpy(labels[i:i + len(imgs)]).to(dev)
        correct += (out.argmax(1) == y).sum().item()
        top5 += (out.topk(5, 1).indices == y[:, None]).any(1).sum().item()
    return correct / n, top5 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-npz", type=str, default=None, help="ADM VIRTUAL_imagenet256_labeled.npz")
    ap.add_argument("--ref-cache", type=str, required=True, help="npz cache of ref Inception features")
    ap.add_argument("--samples", nargs="*", default=[], help="sample folders (expects {folder}_labels.npy)")
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=None, help="evaluate only the first max-n indices")
    args = ap.parse_args()
    fe = get_extractor()

    cache = dict(np.load(args.ref_cache)) if os.path.exists(args.ref_cache) else {}
    if "feats" not in cache or "mu" not in cache:
        assert args.ref_npz, "need --ref-npz to build cache"
        d = np.load(args.ref_npz)
        print("ref npz arrays:", d.files, flush=True)
        if "feats" not in cache:
            t = time.time()
            f, _ = feats_from_uint8(fe, npz_batches(d))
            cache["feats"] = f.astype(np.float32)
            print(f"ref image features {f.shape} in {time.time()-t:.0f}s", flush=True)
        if "mu" in d.files and "sigma" in d.files:  # ADM precomputed full-reference stats (what the ADM evaluator uses for FID)
            cache["mu"], cache["sigma"] = d["mu"], d["sigma"]
            print("using precomputed mu/sigma from ref npz:", d["mu"].shape, d["sigma"].shape, flush=True)
        else:
            cache["mu"], cache["sigma"] = cache["feats"].astype(np.float64).mean(0), np.cov(cache["feats"].astype(np.float64), rowvar=False)
            print("WARNING: no mu/sigma in ref npz; using stats of the reference images only", flush=True)
        np.savez(args.ref_cache, **cache)
    ref_feats = cache["feats"].astype(np.float64)
    ref = RefStats(ref_feats, cache["mu"], cache["sigma"])
    print(f"ref: {len(ref_feats)} image feats for KID; FID vs mu/sigma of shape {ref.sigma.shape}", flush=True)
    rng = np.random.default_rng(args.seed)

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for folder in args.samples:
        folder = folder.rstrip("/")
        idx = available_indices(folder)
        if args.max_n is not None:
            idx = [i for i in idx if i < args.max_n]
        n = len(idx)
        if n == 0:
            print("skip (empty)", folder); continue
        labels = load_labels(folder, max(idx) + 1)
        labels = labels[idx] if labels is not None else None
        t = time.time()
        f, lg = feats_from_uint8(fe, folder_batches(folder, idx))
        r = {"N": n, "FID_N": fid_from_feats(ref, f)}
        r["FID_inf"], r["FID_inf_slope"], r["FID_curve"] = fid_inf(ref, f, rng)
        r["KID_mean"], r["KID_std"] = kid(ref_feats, f, rng)
        r["IS"], _ = inception_score(lg, splits=1)
        # bootstrap over samples for FID_N CI (captures sampling noise of the generated set at this N)
        boots = [fid_from_feats(ref, f[rng.choice(n, n, replace=True)]) for _ in range(10)]
        r["FID_N_boot_std"] = float(np.std(boots))
        if labels is not None and (labels >= 0).all():
            r["top1"], r["top5"] = classifier_acc(folder, idx, labels)
        r["eval_seconds"] = time.time() - t
        results[os.path.basename(folder.rstrip('/').replace('/pngs','')) + (f"_n{n}" if args.max_n else "")] = r
        print(json.dumps({k: v for k, v in r.items() if k != "FID_curve"}, indent=1), flush=True)
        json.dump(results, open(args.out, "w"), indent=1)

    print("\n| arm | N | FID_N (boot sd) | FID_inf | KID x1e3 | IS | top-1 | top-5 |\n|-|-|-|-|-|-|-|-|")
    for k, r in results.items():
        print(f"| {k} | {r['N']} | {r['FID_N']:.2f} ({r['FID_N_boot_std']:.2f}) | {r['FID_inf']:.2f} | "
              f"{1e3*r['KID_mean']:.2f} ± {1e3*r['KID_std']:.2f} | {r['IS']:.1f} | "
              f"{r.get('top1', float('nan')):.3f} | {r.get('top5', float('nan')):.3f} |")


if __name__ == "__main__":
    main()
