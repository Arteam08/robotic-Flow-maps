"""Score npz sample files with the published-scale ("ADM") evaluator; write adm_fid.json next to each.

Portable version of ~/eqm_repro/eqfm_fid/score_npz.py (Babel).  Input: files written by
`scripts/eval_fid_stage2_flowmap.py --save-samples-npz` (key `arr_0`, NHWC uint8, labels = index % 1000).

    python tools/fid_adm/score_npz.py --ref-cache ref_inception.npz samples.npz [more.npz ...]

Build the reference cache once from the ADM reference batch (see tools/fid_adm/README.md):

    python tools/fid_adm/eval_fid.py --ref-npz VIRTUAL_imagenet256_labeled.npz --ref-cache ref_inception.npz

Metrics: FID at N, bootstrap sd, FID_inf (1/N extrapolation), KID, Inception Score, ResNet-50 top-1 vs the
requested labels.  Numbers are comparable ONLY with other numbers from this evaluator at the same N.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_fid import get_extractor, feats_from_uint8, RefStats, fid_from_feats, fid_inf, kid, inception_score

p = argparse.ArgumentParser()
p.add_argument("--ref-cache", default=os.environ.get("EQFM_REF_INCEPTION"),
               help="npz cache of reference Inception features (built by eval_fid.py --ref-npz ...); "
                    "defaults to $EQFM_REF_INCEPTION")
p.add_argument("--num-classes", type=int, default=1000, help="labels are index %% num_classes")
p.add_argument("--no-top1", action="store_true", help="skip the ResNet-50 top-1 check")
p.add_argument("paths", nargs="+")
a = p.parse_args()
if not a.ref_cache:
    p.error("--ref-cache (or $EQFM_REF_INCEPTION) is required")

c = dict(np.load(a.ref_cache))
ref = RefStats(c["feats"].astype(np.float64), c["mu"], c["sigma"])
fe = get_extractor()
rng = np.random.default_rng(0)
if not a.no_top1:
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    net = torchvision.models.resnet50(weights=w).cuda().eval()
    tf = w.transforms()

for path in a.paths:
    t = time.time()
    arr = np.load(path)["arr_0"]
    n = len(arr)
    labels = np.arange(n) % a.num_classes
    f, lg = feats_from_uint8(fe, (arr[i:i + 250] for i in range(0, n, 250)))
    r = {"path": path, "N": n, "FID_N": fid_from_feats(ref, f)}
    r["FID_inf"], r["FID_inf_slope"], _ = fid_inf(ref, f, rng)
    r["KID_mean"], r["KID_std"] = kid(ref.feats, f, rng)
    r["IS"], _ = inception_score(lg)
    r["FID_N_boot_std"] = float(np.std([fid_from_feats(ref, f[rng.choice(n, n, replace=True)]) for _ in range(10)]))
    if n >= 2048:
        r["FID_2048_first"] = fid_from_feats(ref, f[:2048])
    if not a.no_top1:
        correct = 0
        with torch.no_grad():
            for i in range(0, n, 128):
                x = torch.stack([tf(Image.fromarray(im)) for im in arr[i:i + 128]]).cuda()
                correct += (net(x).argmax(1).cpu().numpy() == labels[i:i + 128]).sum()
        r["top1"] = float(correct / n)
    r["evaluator"] = "torch-fidelity TF-Inception, ADM VIRTUAL_imagenet256 full mu/sigma"
    r["eval_seconds"] = time.time() - t
    out = os.path.join(os.path.dirname(os.path.abspath(path)), "adm_fid.json")
    json.dump(r, open(out, "w"), indent=1)
    print(f"N={n} FID_N={r['FID_N']:.2f} ({r['FID_N_boot_std']:.2f}) FID_inf={r['FID_inf']:.2f} "
          f"KID={1e3 * r['KID_mean']:.2f}e-3 IS={r['IS']:.1f} top1={r.get('top1', float('nan')):.3f} -> {out}", flush=True)
