#!/usr/bin/env python3
"""
compare_runs.py — turn a CPU reference run and an NPU replay into the CPU-vs-NPU
comparison table for the blog.

Inputs
------
  --cpu     the JSONL written live by `live_speed.py --log` on the CPU. Carries
            raw_kmh (the reference prediction), infer_ms, and — crucially — the
            per-clip panel speed set_kmh (ground truth from the treadmill screen).
  --npu     the JSONL written by `replay_clips.py --provider qnn` — the SAME clips
            replayed through the NPU. Carries raw_kmh + infer_ms per clip_id.

Because both are keyed by clip_id and the NPU replayed the exact tensors the CPU
logged, the comparison isolates the engine: no walk-to-walk variation leaks in.

Outputs a Markdown report (and optional --json) with:
  * numeric fidelity : MAE / max abs diff of NPU raw_kmh vs CPU raw_kmh
  * accuracy vs truth: MAE of each engine vs the panel set_kmh, per speed band
  * performance      : p50 / p95 / mean infer_ms for each engine, and speedup
"""

import argparse
import json
import statistics as st


def load(path):
    header, rows = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("type") == "header":
                header = o
            elif o.get("type") == "infer":
                rows[o["clip_id"]] = o
    return header, rows


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def mae(pairs):
    return sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else float("nan")


def band(v):
    # 0-1, 1-2, ... km/h bands for per-speed accuracy
    lo = int(v)
    return f"{lo}-{lo+1}"


def main():
    ap = argparse.ArgumentParser(description="CPU-vs-NPU comparison report")
    ap.add_argument("--cpu", required=True, help="live_speed.py --log JSONL (CPU)")
    ap.add_argument("--npu", required=True, help="replay_clips.py JSONL (NPU)")
    ap.add_argument("--out", default=None, help="write Markdown here (else stdout)")
    ap.add_argument("--json", default=None, help="also dump raw metrics as JSON")
    args = ap.parse_args()

    cpu_h, cpu = load(args.cpu)
    npu_h, npu = load(args.npu)
    common = sorted(set(cpu) & set(npu))
    if not common:
        raise SystemExit("no overlapping clip_id between the two runs")

    # ── numeric fidelity: NPU vs CPU on identical clips ──────────────────────
    diffs = [(cpu[c]["raw_kmh"], npu[c]["raw_kmh"]) for c in common]
    fidelity_mae = mae(diffs)
    max_abs = max(abs(a - b) for a, b in diffs)

    # ── accuracy vs panel ground truth (only clips that carry set_kmh) ───────
    def acc_vs_truth(src, key):
        pairs, per_band = [], {}
        for c in common:
            gt = cpu[c].get("set_kmh")
            if gt is None:
                continue
            pred = src[c]["raw_kmh"]
            pairs.append((pred, gt))
            per_band.setdefault(band(gt), []).append(abs(pred - gt))
        overall = mae(pairs)
        bands = {b: sum(v) / len(v) for b, v in sorted(per_band.items())}
        return overall, bands

    cpu_acc, cpu_bands = acc_vs_truth(cpu, "cpu")
    npu_acc, npu_bands = acc_vs_truth(npu, "npu")

    # ── performance ──────────────────────────────────────────────────────────
    def perf(src):
        ms = [src[c]["infer_ms"] for c in common]
        return {"mean": st.mean(ms), "p50": pct(ms, 50), "p95": pct(ms, 95),
                "min": min(ms), "max": max(ms)}

    cpu_p, npu_p = perf(cpu), perf(npu)
    speedup = cpu_p["p50"] / npu_p["p50"] if npu_p["p50"] else float("nan")

    # ── render ─────────────────────────────────────────────────────────────
    L = []
    L.append("# CPU vs NPU — treadmill speed model\n")
    L.append(f"- clips compared: **{len(common)}**")
    L.append(f"- CPU log: `{args.cpu}` (provider={cpu_h.get('provider','?')}, "
             f"model={cpu_h.get('model','?')})")
    L.append(f"- NPU log: `{args.npu}` (provider={npu_h.get('provider','?')})")
    if cpu_h.get("model_sha256"):
        L.append(f"- model sha256: `{cpu_h['model_sha256'][:16]}…`")
    L.append("")
    L.append("## 1. Numeric fidelity (NPU vs CPU, identical clips)\n")
    L.append(f"- **MAE(NPU − CPU) = {fidelity_mae:.4f} km/h**")
    L.append(f"- max |NPU − CPU|  = {max_abs:.4f} km/h\n")
    L.append("_This is the quantization cost: how far A16W8 on the NPU drifts "
             "from the float CPU output on the very same input._\n")
    L.append("## 2. Accuracy vs panel ground truth\n")
    L.append("| engine | MAE vs panel (km/h) |")
    L.append("|---|---|")
    L.append(f"| CPU (float) | {cpu_acc:.3f} |")
    L.append(f"| NPU (A16W8) | {npu_acc:.3f} |")
    L.append("")
    if cpu_bands:
        L.append("### Per speed band (MAE vs panel, km/h)\n")
        L.append("| band km/h | CPU | NPU |")
        L.append("|---|---|---|")
        for b in sorted(set(cpu_bands) | set(npu_bands),
                        key=lambda x: int(x.split("-")[0])):
            L.append(f"| {b} | {cpu_bands.get(b, float('nan')):.3f} "
                     f"| {npu_bands.get(b, float('nan')):.3f} |")
        L.append("")
    L.append("## 3. Performance (per-inference latency)\n")
    L.append("| engine | mean ms | p50 ms | p95 ms | min | max |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| CPU | {cpu_p['mean']:.2f} | {cpu_p['p50']:.2f} | {cpu_p['p95']:.2f} "
             f"| {cpu_p['min']:.2f} | {cpu_p['max']:.2f} |")
    L.append(f"| NPU | {npu_p['mean']:.2f} | {npu_p['p50']:.2f} | {npu_p['p95']:.2f} "
             f"| {npu_p['min']:.2f} | {npu_p['max']:.2f} |")
    L.append("")
    L.append(f"- **p50 speedup (CPU/NPU): {speedup:.1f}×**")
    L.append("")
    report = "\n".join(L)

    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"wrote {args.out}")
    else:
        print(report)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"n": len(common), "fidelity_mae": fidelity_mae,
                       "max_abs_diff": max_abs, "cpu_acc": cpu_acc,
                       "npu_acc": npu_acc, "cpu_bands": cpu_bands,
                       "npu_bands": npu_bands, "cpu_perf": cpu_p,
                       "npu_perf": npu_p, "p50_speedup": speedup}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
