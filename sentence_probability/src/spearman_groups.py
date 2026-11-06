from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

GROUPS = ("EC–EC", "NEC–NEC", "EC–NEC")


def pair_group(a_ec, b_ec):
    if a_ec and b_ec:
        return "EC–EC"
    if not a_ec and not b_ec:
        return "NEC–NEC"
    return "EC–NEC"


def pair_table(rho, ec, family, labels=None):
    names = list(rho.index)
    labels = labels or {}
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({"model_a": a, "model_b": b, "label_a": labels.get(a, a), "label_b": labels.get(b, b),
                         "family_a": family[a], "family_b": family[b],
                         "group": pair_group(ec[a], ec[b]), "same_family": family[a] == family[b],
                         "rho": float(rho.loc[a, b])})
    return pd.DataFrame(rows)


def describe(values):
    v = np.asarray(values, dtype=float)
    ok = v[np.isfinite(v)]
    out = {"n_pairs": int(len(v)), "n_pairs_defined": int(len(ok)), "n_pairs_missing": int(len(v) - len(ok))}
    if not len(ok):
        return {**out, **{k: np.nan for k in ("median", "mean", "sd", "q1", "q3", "iqr", "min", "max")}}
    q1, q3 = np.percentile(ok, [25, 75])
    return {**out, "median": float(np.median(ok)), "mean": float(ok.mean()),
            "sd": float(ok.std(ddof=1)) if len(ok) > 1 else np.nan,
            "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1), "min": float(ok.min()), "max": float(ok.max())}


def group_summary(pairs, ec):
    rows = []
    n_ec = sum(bool(v) for v in ec.values())
    n_nec = len(ec) - n_ec
    n_models = {"EC–EC": n_ec, "NEC–NEC": n_nec, "EC–NEC": n_ec + n_nec}
    expected = {"EC–EC": n_ec * (n_ec - 1) // 2, "NEC–NEC": n_nec * (n_nec - 1) // 2, "EC–NEC": n_ec * n_nec}
    for g in GROUPS:
        sub = pairs.loc[pairs["group"] == g, "rho"]
        d = describe(sub)
        assert d["n_pairs"] == expected[g], (g, d["n_pairs"], expected[g])
        rows.append({"group": g, "role": "descriptive reference" if g == "EC–NEC" else "compared",
                     "n_models": n_models[g], **d})
    return pd.DataFrame(rows)


def cross_family_summary(pairs):
    rows = []
    for g in ("EC–EC", "NEC–NEC", "EC–NEC"):
        for same, name in ((True, "same family"), (False, "different families")):
            sub = pairs.loc[(pairs["group"] == g) & (pairs["same_family"] == same)]
            fams = sorted(set(sub["family_a"]) | set(sub["family_b"]))
            d = describe(sub["rho"])
            rows.append({"group": g, "family_relation": name, "category": f"{g}, {name}",
                         "n_families_involved": len(fams), "families": ", ".join(fams),
                         "available": d["n_pairs"] > 0, **d})
    return pd.DataFrame(rows)


def _upper(k):
    return np.triu_indices(k, 1)


def delta_median(r, ec_mask, pair_mask=None):
    iu, ju = _upper(len(ec_mask))
    vals = r[iu, ju]
    keep = np.isfinite(vals) if pair_mask is None else np.isfinite(vals) & pair_mask[iu, ju]
    e = ec_mask[iu] & ec_mask[ju] & keep
    n = ~ec_mask[iu] & ~ec_mask[ju] & keep
    if not e.any() or not n.any():
        return np.nan
    return float(np.median(vals[e]) - np.median(vals[n]))


def family_balanced_delta(r, ec_mask, fam_codes):
    fams = np.unique(fam_codes)
    cells = {"EC–EC": [], "NEC–NEC": []}
    for fa, fb in itertools.combinations(fams, 2):
        ia, ib = np.flatnonzero(fam_codes == fa), np.flatnonzero(fam_codes == fb)
        ea, eb = ec_mask[ia].all(), ec_mask[ib].all()
        if ea != eb:
            continue
        v = r[np.ix_(ia, ib)]
        v = v[np.isfinite(v)]
        if len(v):
            cells["EC–EC" if ea else "NEC–NEC"].append(float(v.mean()))
    e, n = cells["EC–EC"], cells["NEC–NEC"]
    d = float(np.median(e) - np.median(n)) if e and n else np.nan
    return d, len(e), len(n), (float(np.median(e)) if e else np.nan), (float(np.median(n)) if n else np.nan)


def leave_one_out(r, ec_mask, keys, labels):
    out = []
    for k_, lab in zip(keys, labels):
        keep = ~k_
        sub = r[np.ix_(keep, keep)]
        out.append({"dropped": lab, "n_ec": int(ec_mask[keep].sum()), "n_nec": int((~ec_mask[keep]).sum()),
                    "delta_median_rho": delta_median(sub, ec_mask[keep])})
    return pd.DataFrame(out)


def run_group_comparison(rho, ec, family, labels=None, family_alt=None):
    names = list(rho.index)
    r = rho.to_numpy(dtype=float)
    assert np.allclose(r, r.T, equal_nan=True), "rho matrix not symmetric"
    ec_mask = np.array([bool(ec[m]) for m in names])
    fam = np.array([family[m] for m in names], dtype=object)
    _u, fam_codes = np.unique(fam.astype(str), return_inverse=True)
    pairs = pair_table(rho, ec, family, labels)
    summary = group_summary(pairs, ec)
    cross = cross_family_summary(pairs)
    observed = delta_median(r, ec_mask)
    med = summary.set_index("group")["median"]
    assert math.isclose(observed, med["EC–EC"] - med["NEC–NEC"], abs_tol=1e-12) or not np.isfinite(observed)
    res = {"names": names, "ec_mask": ec_mask, "family": fam, "pairs": pairs, "summary": summary, "cross": cross,
           "observed": observed}
    cross_mask = fam[:, None] != fam[None, :]
    res["cross_family_delta"] = delta_median(r, ec_mask, cross_mask)
    res["family_balanced"] = family_balanced_delta(r, ec_mask, fam_codes)
    fam_names = list(_u)
    res["lofo"] = leave_one_out(r, ec_mask, [fam_codes == i for i in range(len(fam_names))], fam_names)
    res["lomo"] = leave_one_out(r, ec_mask, [np.arange(len(names)) == i for i in range(len(names))],
                                [(labels or {}).get(m, m) for m in names])
    if family_alt is not None:
        alt = np.array([family_alt[m] for m in names], dtype=object)
        _ua, alt_codes = np.unique(alt.astype(str), return_inverse=True)
        alt_cross = alt[:, None] != alt[None, :]
        res["alt_family"] = {"n_families": len(_ua), "cross_family_delta": delta_median(r, ec_mask, alt_cross),
                             "family_balanced": family_balanced_delta(r, ec_mask, alt_codes),
                             "cross": cross_family_summary(pair_table(rho, ec, family_alt, labels))}
    return res


def ec_nec_masks(ec_mask, pair_mask=None):
    iu, ju = _upper(len(ec_mask))
    keep = np.ones(len(iu), dtype=bool) if pair_mask is None else pair_mask[iu, ju]
    ee = ec_mask[iu] & ec_mask[ju] & keep
    en = (ec_mask[iu] != ec_mask[ju]) & keep
    nn = ~ec_mask[iu] & ~ec_mask[ju] & keep
    return iu, ju, ee, en, nn


def delta_ec_vs_ecnec(r, ec_mask, pair_mask=None):
    iu, ju, ee, en, _nn = ec_nec_masks(ec_mask, pair_mask)
    v = r[iu, ju]
    fin = np.isfinite(v)
    a, b = v[ee & fin], v[en & fin]
    ma = float(np.median(a)) if len(a) else np.nan
    mb = float(np.median(b)) if len(b) else np.nan
    return ma, mb, ma - mb, int(len(a)), int(len(b))


def ec_vs_ecnec_loo(r, ec_mask, keys, labels):
    out = []
    for k_, lab in zip(keys, labels):
        keep = ~k_
        ma, mb, d, na, nb = delta_ec_vs_ecnec(r[np.ix_(keep, keep)], ec_mask[keep])
        out.append({"dropped": lab, "n_ec": int(ec_mask[keep].sum()), "n_nec": int((~ec_mask[keep]).sum()),
                    "n_ec_ec_pairs": na, "n_ec_nec_pairs": nb, "median_rho_ec_ec": ma, "median_rho_ec_nec": mb,
                    "delta_median_rho": d})
    return pd.DataFrame(out)
