"""Confirmatory analyses of the frozen plan (PREREGISTRATION.md). Every number in the paper comes from here.

    python scripts/12_analysis.py rq3        # judge validity against EPITOME human labels
    python scripts/12_analysis.py prepare    # long table reply x judge + objective measures, for R
    python scripts/12_analysis.py judges     # inter-judge alpha per variety; judge emotion/urgency vs gold
    python scripts/12_analysis.py natural    # human-written vs translated Darija (RQ1, ecological validity)
Outputs go to results/analysis/.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis"
JUDGE_DIR = ROOT / "results/judge"
N_BOOT = 2000
SEED = 2026
DIMS = ["er", "ip", "ex"]
VARIETIES = ["en", "fr", "msa", "ary", "ary_latn"]


def qwk(a, b, k=3):
    """Quadratic-weighted Cohen's kappa for labels 0..k-1 (same value as sklearn, vectorised for bootstrap)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    o = np.zeros((k, k))
    np.add.at(o, (a, b), 1)
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / max(1, o.sum())
    den = (w * e).sum()
    return 1 - (w * o).sum() / den if den else np.nan


def holm(p):
    """Holm-adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


APPROP = {"too_little": -1, "appropriate": 0, "excessive": 1}


def prepare():
    """results/analysis/replies_long.csv (one row per reply x judge) and replies_objective.csv."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from langid_darija import identify
    from prompts import SAFETY_PATTERNS
    judge11 = __import__("importlib").import_module("11_judge")
    OUT.mkdir(parents=True, exist_ok=True)
    rep = judge11.load_replies()
    seeds = pd.read_csv(ROOT / "data/processed/scenarios_seed.csv")[["scenario_id", "emotion_target"]]
    rep = rep.merge(seeds, on="scenario_id", how="left")

    obj_path = OUT / "replies_objective.csv"
    cached = pd.read_csv(obj_path) if obj_path.exists() else pd.DataFrame(columns=["item_key", "reply_lid"])
    lid = dict(zip(cached.item_key, cached.reply_lid))
    todo = rep[~rep.item_key.isin(lid)]
    lid.update({k: identify(t)[0] for k, t in zip(todo.item_key, todo.reply)})
    rep["reply_lid"] = rep.item_key.map(lid)
    rep["fidelity"] = (rep.reply_lid == rep.variety).astype(int)
    rep["safety_regex"] = rep.reply.apply(lambda t: int(any(re.search(p, str(t), re.I) for p in SAFETY_PATTERNS)))
    rep["words"] = rep.reply.str.split().str.len()
    rep["family"] = np.where(rep.system.str.startswith("claude"), "claude", "other")
    rep.drop(columns=["reply", "message"]).to_csv(obj_path, index=False)

    rows = []
    for f in sorted(JUDGE_DIR.glob("*_replies.csv")):
        j = pd.read_csv(f)
        rows.append(j[j.parse_ok])
    if not rows:
        print("no judge files yet")
        return
    j = pd.concat(rows, ignore_index=True)
    for c in ["er", "ip", "ex", "empathy", "helpfulness", "naturalness"]:
        j[c] = pd.to_numeric(j[c], errors="coerce")
    j["approp"] = j.appropriateness.map(APPROP)
    j["action_cov"] = j.actions.apply(lambda a: np.mean([bool(x) for x in json.loads(a)]) if isinstance(a, str) and a.startswith("[") and a != "[]" else np.nan)
    j["safety_judge"] = j.safety_violation.astype(str).str.lower().isin(["true", "1"]).astype(int)
    j["judge_family"] = np.where(j.judge.isin(["haiku", "sonnet"]), "claude", "other")
    keep = ["item_key", "judge", "judge_family", "emotion", "urgency", "er", "ip", "ex", "empathy", "approp",
            "appropriateness", "helpfulness", "naturalness", "action_cov", "safety_judge"]
    long = rep.drop(columns=["reply", "message"]).merge(j[keep].rename(columns={"emotion": "judge_emotion",
                                                                              "urgency": "judge_urgency"}), on="item_key")
    long["safety"] = long.safety_regex * long.safety_judge
    long.to_csv(OUT / "replies_long.csv", index=False)
    print(f"{len(rep)} replies, {len(long)} reply x judge rows; judges: {long.judge.value_counts().to_dict()}")
    print(long.groupby(["system", "policy"]).agg(fidelity=("fidelity", "mean"), empathy=("empathy", "mean"),
                                                  appropriate=("approp", lambda x: (x == 0).mean()),
                                                  helpful=("helpfulness", "mean")).round(2).to_string())


def judges():
    """Inter-judge Krippendorff alpha per variety and measure (benchmark), and judge estimates of the
    customer's emotion and urgency against the gold labels."""
    import krippendorff
    d = pd.read_csv(OUT / "replies_long.csv")
    b = d[d.set == "benchmark"]
    rows = []
    for v, g in b.groupby("variety"):
        for meas, level in (("empathy", "ordinal"), ("helpfulness", "ordinal"), ("approp", "ordinal"),
                            ("er", "ordinal"), ("ip", "ordinal"), ("ex", "ordinal")):
            piv = g.pivot_table(index="judge", columns="item_key", values=meas, aggfunc="first")
            if len(piv) < 2:
                continue
            a = krippendorff.alpha(reliability_data=piv.to_numpy(dtype=float), level_of_measurement=level)
            rows.append({"variety": v, "measure": meas, "judges": len(piv), "items": piv.shape[1], "alpha": a})
    al = pd.DataFrame(rows)
    al.to_csv(OUT / "rq3_interjudge_alpha.csv", index=False)
    if len(al):
        print(al.pivot_table(index="measure", columns="variety", values="alpha").round(3).to_string())
    # Judge emotion / urgency estimates vs gold (benchmark, one row per reply x judge).
    acc = b.assign(emo_ok=b.judge_emotion == b.emotion_target, urg_ok=b.judge_urgency == b.urgency) \
        .groupby(["judge", "variety"])[["emo_ok", "urg_ok"]].mean()
    acc.to_csv(OUT / "judge_label_accuracy.csv")
    print(acc.round(3).unstack().to_string())


def natural():
    """Per system and policy: jury scores and fidelity on the 120 human-written Darija messages vs. the 152
    translated Darija (Arabic-script) benchmark messages; bootstrap CI of the difference over messages."""
    d = pd.read_csv(OUT / "replies_long.csv")
    d = d[(d.variety == "ary") & d.policy.isin(["P0", "P1", "P2", "none"])]
    per_reply = d.groupby(["item_key", "set", "system", "policy", "item_id"]).agg(
        empathy=("empathy", "median"), helpfulness=("helpfulness", "median"),
        appropriate=("approp", lambda x: float(np.median(x) == 0)), fidelity=("fidelity", "first")).reset_index()
    rng = np.random.default_rng(SEED)
    rows = []
    for (sy, pol), g in per_reply.groupby(["system", "policy"]):
        nat, ben = g[g.set == "natural"], g[g.set == "benchmark"]
        if len(nat) < 20 or len(ben) < 20:
            continue
        for m in ("fidelity", "empathy", "helpfulness", "appropriate"):
            a, b = nat[m].dropna().to_numpy(), ben[m].dropna().to_numpy()
            diff = a.mean() - b.mean()
            bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)]
            rows.append({"system": sy, "policy": pol, "measure": m, "natural": a.mean(), "translated": b.mean(),
                         "diff": diff, "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5),
                         "n_natural": len(a), "n_translated": len(b)})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "natural_vs_translated.csv", index=False)
    print(r[r.policy.isin(["P2", "none"])].round(3).to_string(index=False))


def load_epitome_scores():
    gold = pd.read_csv(ROOT / "data/processed/epitome_final.csv")
    gold["item_key"] = gold.epitome_id.astype(str) + "|" + gold.variety
    frames = []
    for f in sorted(JUDGE_DIR.glob("*_epitome.csv")):
        j = pd.read_csv(f)
        j = j[j.parse_ok].copy()
        for d in DIMS:
            j[d] = pd.to_numeric(j[d], errors="coerce").clip(0, 2)
        frames.append(j.dropna(subset=DIMS))
    judged = pd.concat(frames, ignore_index=True)
    # The jury: per item and mechanism, the median of the judges; halves are rounded down (conservative).
    jury = judged.groupby("item_key")[DIMS].median().apply(np.floor).astype(int).reset_index()
    judged = pd.concat([judged, jury.assign(judge="jury")], ignore_index=True)
    m = judged.merge(gold[["item_key", "epitome_id", "variety", "qc_pass"] + DIMS], on="item_key",
                     suffixes=("_judge", "_human"))
    return m


def rq3():
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_epitome_scores()
    rng = np.random.default_rng(SEED)
    ids = np.array(sorted(m.epitome_id.unique()))
    boot_ids = [rng.choice(ids, len(ids), replace=True) for _ in range(N_BOOT)]
    rows, drops = [], []
    for subset, mm in (("all", m), ("qc_pass", m[m.qc_pass.astype(bool)])):
        for (judge, variety), g in mm.groupby(["judge", "variety"]):
            g = g.set_index("epitome_id")
            for d in DIMS + ["total"]:
                h = g[[f"{x}_human" for x in DIMS]].sum(1) if d == "total" else g[f"{d}_human"]
                p = g[[f"{x}_judge" for x in DIMS]].sum(1) if d == "total" else g[f"{d}_judge"]
                stat = (lambda a, b: spearmanr(a, b).statistic) if d == "total" else qwk
                est = stat(h, p)
                hv, pv = h.to_numpy(), p.to_numpy()
                bs = []
                for b in boot_ids:
                    pos = g.index.get_indexer(b)
                    pos = pos[pos >= 0]
                    bs.append(stat(hv[pos], pv[pos]))
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                rows.append({"subset": subset, "judge": judge, "variety": variety, "dimension": d,
                             "metric": "spearman" if d == "total" else "qwk", "n": len(g),
                             "estimate": est, "ci_low": lo, "ci_high": hi})
        # Paired bootstrap: drop from en to each variety, on total empathy (Spearman), per judge.
        for judge, gj in mm.groupby("judge"):
            tot = gj.assign(h=gj[[f"{x}_human" for x in DIMS]].sum(1), p=gj[[f"{x}_judge" for x in DIMS]].sum(1))
            piv_h = tot.pivot_table(index="epitome_id", columns="variety", values="h")
            piv_p = tot.pivot_table(index="epitome_id", columns="variety", values="p")
            for v in VARIETIES[1:]:
                both = piv_h[["en", v]].dropna().index.intersection(piv_p[["en", v]].dropna().index)
                if len(both) < 20:
                    continue
                he, pe = piv_h.loc[both, "en"].to_numpy(), piv_p.loc[both, "en"].to_numpy()
                hvv, pvv = piv_h.loc[both, v].to_numpy(), piv_p.loc[both, v].to_numpy()
                f = lambda pos: spearmanr(he[pos], pe[pos]).statistic - spearmanr(hvv[pos], pvv[pos]).statistic
                est = f(np.arange(len(both)))
                bs = []
                for b in boot_ids:
                    pos = both.get_indexer(b)
                    bs.append(f(pos[pos >= 0]))
                bs = np.array(bs)
                drops.append({"subset": subset, "judge": judge, "variety": v, "n": len(both),
                              "drop_en_minus_v": est, "ci_low": np.nanpercentile(bs, 2.5),
                              "ci_high": np.nanpercentile(bs, 97.5), "p_one_sided": float(np.mean(bs <= 0))})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "rq3_agreement.csv", index=False)
    dr = pd.DataFrame(drops)
    if len(dr):
        dr["p_holm"] = dr.groupby("subset").p_one_sided.transform(holm)
        dr.to_csv(OUT / "rq3_drop_vs_en.csv", index=False)
    print(res[(res.subset == "all") & (res.dimension == "total")]
          .pivot_table(index="judge", columns="variety", values="estimate")[VARIETIES].round(3).to_string())
    if len(dr):
        print(dr[dr.subset == "all"].round(3).to_string(index=False))


if __name__ == "__main__":
    {"rq3": rq3, "prepare": prepare, "judges": judges, "natural": natural}[sys.argv[1]]()
