"""CPU checks for readable wandb logging (run: python tests/test_wandb_glossary.py; no pytest needed)."""
import sys, types, math, tempfile, json
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
torch.set_num_threads(2)

from eqfm import wandb_glossary as wg
from eqfm.metric_window import WindowStats, pool_eqm_metrics
from eqfm.losses import EqMLoss, EqMLossConfig


def args_ns(**kw):
    d = dict(time_sampler="uniform_weighted", eps_train=1e-3, data_anchor=0.5, time_pred_weight=0.0,
             parameterization="velocity", ema_decay=0.9999, grad_clip=1.0, cfg_dropout_prob=0.1,
             warmup_steps=500, val_num_steps=100, val_eps_stop=1e-3, val_cfg_scale=1.5, val_n_samples=16,
             val_loss_batches=4, batch_size=32, ckpt="EqM.pt", ckpt_key="model", init_output_scale="1",
             trainable=None, time_conditioning="zero", lr=3e-5, grad_accum_steps=4, bf16=True,
             shards="parquet:x", hflip=True, val_every=2500, log_every=50, results_dir="/r", run_name="demo",
             init_from_scratch=False)
    d.update(kw); return types.SimpleNamespace(**d)


def test_stage1_keys_resolve_and_are_unique():
    ctx = wg.stage1_context(args_ns(), world_size=2)
    torch.manual_seed(0)
    loss_fn = EqMLoss(EqMLossConfig(eps_train=1e-3, time_sampler="uniform_weighted", data_anchor_weight=0.5))
    x = torch.randn(64, 4, 8, 8); y = torch.randint(0, 1000, (64,))
    fn = lambda z, yy: 0.3 * z
    loss, metrics = loss_fn(fn, x, y)
    st = WindowStats()
    pool_eqm_metrics(st, metrics, drop_mask=torch.rand(64) < 0.3)
    st.add_scalar("cfg_drop_frac", 0.1); st.add_scalar("grad_norm", 1.0); st.add_scalar("grad_norm_max", 2.0)
    res = st.result()
    trainer_keys = set(res) | {"lr", "clip_frac", "images_per_s", "sec_per_step", "data_wait_frac", "images_seen",
                               "epoch", "param_norm", "ema_gap", "gpu_mem_gb", "init_gap",
                               "init/scale", "init/objective", "init/b_data_rms", "init/x1_norm",
                               "samples/grid", "samples/b_norm_first", "samples/b_norm_last", "samples/b_norm_max",
                               "samples/x_norm_last", "samples/cfg_scale"}
    trainer_keys |= {f"probe_{w}/{k}" for w in ("raw", "ema") for k in
                     ["loss", "b_data_rms"] + [k for k in res if k.startswith(("eqm_raw_t", "eqm_relerr_t"))]}
    shown = {}
    for k in sorted(trainer_keys):
        r = wg.resolve(wg.STAGE1, k, ctx)
        assert r is not None, f"unregistered stage-1 key {k}"
        disp = r[0]
        assert disp is not None and "{" not in disp and "{" not in r[1], (k, disp, r[1])
        assert disp not in shown, f"{k} and {shown.get(disp)} both map to {disp!r}"
        shown[disp] = k
        assert len(disp) <= 60, (k, disp, len(disp))
    print(f"stage 1: {len(shown)} keys resolve uniquely; longest = {max(map(len, shown))} chars")
    for d in ["2 loss by t/mse  t∈[0.9, 1]", "2 loss by t/mse  t∈[0, 0.1)", "4 probe/objective  E[w·mse], EMA θ",
              "3 field by t/cos(b, U_t)  all t", "1 loss/mse, null label"]:
        assert d in shown, (d, sorted(shown)[:10])


def test_stage2_legacy_keys_resolve_unique():
    live = """train/cfg_drop_frac train/diag_loss train/diag_raw train/diag_raw_t0.1_0.5 train/diag_raw_t0.5_0.9 train/diag_raw_t0.9_1
    train/diag_raw_t0_0.1 train/distill_loss train/distill_raw train/distill_raw_max train/distill_raw_s0.5_0.9 train/distill_raw_s0.99_1
    train/distill_raw_s0.9_0.99 train/distill_raw_s0_0.5 train/distill_raw_t0.1_0.5 train/distill_raw_t0.5_0.9 train/distill_raw_t0.9_1
    train/distill_raw_t0_0.1 train/effective_global_batch train/grad_accum_steps train/grad_norm train/grad_skips train/interp_t_mean
    train/lag_loss train/loss train/lr train/samples_per_s train/semi_loss train/semi_raw train/semi_raw_s0.5_0.9 train/semi_raw_s0.99_1
    train/semi_raw_s0.9_0.99 train/semi_raw_s0_0.5 train/sigma_mean train/terminal_invariance_endpoint_delta train/terminal_invariance_loss
    train/terminal_loss train/terminal_student_endpoint_norm train/terminal_student_norm_loss train/terminal_teacher_endpoint_norm
    train/terminal_teacher_norm_loss val/diag_euler_samples val/diag_latent_norm_last val/diag_velocity_norm_first
    val/diag_velocity_norm_last val/endpoint_b_norm val/map_err_s0.500 val/map_err_s0.900 val/map_err_s0.970 val/map_err_s0.999
    val/offdiag_K1_samples val/offdiag_K2_samples val/offdiag_K4_samples val/offdiag_K8_samples val/terminal_samples""".split()
    shown = {}
    for k in live:
        r = wg.resolve(wg.STAGE2, k, {})
        assert r is not None, f"unregistered stage-2 key {k}"
        assert "{" not in r[0], (k, r[0])
        assert r[0] not in shown, f"{k} and {shown.get(r[0])} both map to {r[0]!r}"
        shown[r[0]] = k
    print(f"stage 2: {len(shown)} live keys resolve uniquely, e.g. {wg.resolve(wg.STAGE2, 'val/map_err_s0.970', {})[0]!r}")


def test_window_stats_exact():
    torch.manual_seed(1)
    st = WindowStats()
    ts, vs = [], []
    for _ in range(5):
        t = torch.rand(8); v = torch.rand(8)
        st.add_samples(t, {"q": v}); st.add_share("sh", t, v)
        st.add_scalar("a", v.mean()); st.add_scalar("b_max", v.max())
        ts.append(t); vs.append(v)
    t = torch.cat(ts); v = torch.cat(vs); r = st.result()
    assert math.isclose(r["q"], v.mean().item(), rel_tol=1e-6)
    m = (t >= 0.9) & (t <= 1.0)
    assert math.isclose(r["q_t0.9_1"], v[m].mean().item(), rel_tol=1e-6)
    assert math.isclose(r["sh_t0.9_1"], (v * m).sum().item() / v.sum().item(), rel_tol=1e-6)
    assert math.isclose(sum(r[f"sh_{b}"] for b in ("t0_0.1", "t0.1_0.5", "t0.5_0.9", "t0.9_1")), 1.0, rel_tol=1e-6)
    assert math.isclose(r["b_max"], v.max().item(), rel_tol=1e-6)
    st2 = WindowStats(); st2.add_samples(torch.tensor([0.95, 0.97]), {"q": torch.tensor([1.0, 3.0])})
    r2 = st2.result()
    assert "q_t0_0.1" not in r2 and r2["q_t0.9_1"] == 2.0, r2   # empty bucket omitted, not 0
    print("window stats: pooled means, shares, max, empty-bucket omission OK")


def test_fixed_batch_loss_and_logger_offline():
    from eqfm.eval_utils import fixed_batch_loss
    from eqfm.models.sit import SiT_models
    from eqfm.eval_utils import autonomous_forward
    torch.manual_seed(0)
    m = SiT_models["SiT-S/2"](input_size=8).eval()
    fn = autonomous_forward(m)
    loss_fn = EqMLoss(EqMLossConfig(eps_train=1e-3, time_sampler="uniform_weighted"))
    batches = [(torch.randn(6, 4, 8, 8), torch.randint(0, 1000, (6,))) for _ in range(3)]
    a = fixed_batch_loss(fn, loss_fn, batches, seed=0, device=torch.device("cpu"), bf16=False)
    b = fixed_batch_loss(fn, loss_fn, batches, seed=0, device=torch.device("cpu"), bf16=False)
    assert a == b and {"loss", "b_data_rms", "x1_norm", "eqm_raw", "eqm_relerr"} <= set(a), sorted(a)[:12]
    print(f"fixed_batch_loss deterministic, {len(a)} keys")

    import wandb
    with tempfile.TemporaryDirectory() as d:
        run = wandb.init(project="glossary-test", mode="offline", dir=d)
        logs = []
        wl = wg.ReadableLogger(run, wg.STAGE1, wg.stage1_context(args_ns()), notation=wg.NOTATION_STAGE1,
                               header=wg.stage1_header(args_ns()), run_dir=Path(d), summaries=wg.SUMMARIES_STAGE1,
                               printer=logs.append)
        out = wl.log({"total_loss": torch.tensor(0.05), "eqm_raw_t0.9_1": 0.01, "mystery_metric": 1.0,
                      "_per_sample": torch.ones(3)}, step=1)
        assert "1 loss/total (optimized)" in out and "9 other/mystery_metric" in out and "0 init/glossary" in out
        assert any("mystery_metric" in l for l in logs), logs
        assert run.notes.startswith("# Stage 1") and "## Notation" in run.notes and "2 loss by t/mse" in run.notes
        assert (Path(d) / "wandb_glossary.md").read_text() == run.notes
        run.finish()
    print("ReadableLogger: notes, table, file, unregistered-key fallback OK")


if __name__ == "__main__":
    test_window_stats_exact()
    test_stage1_keys_resolve_and_are_unique()
    test_stage2_legacy_keys_resolve_unique()
    test_fixed_batch_loss_and_logger_offline()
    print("ALL OK")
