"""Decision rule + Markdown report writer."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    diff = a.astype(np.int32) - b.astype(np.int32)
    mse = float((diff * diff).mean())
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def l1_mean(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


def pixel_exact(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def matches_oracle(exact: bool, p: float, psnr_threshold: float = 50.0) -> bool:
    return exact or (not math.isnan(p) and p >= psnr_threshold)


def derive_verdict(rows: list[dict],
                   workflow_array_match_pct: float | None,
                   rgb_decord: str, rgb_pyav_toc: str) -> tuple[str, list[str]]:
    """rows is the per-(clip,reader,index) result list."""
    notes: list[str] = []

    # Bucket by reader+clip+codec
    def clip_codec(name: str) -> str:
        # heuristic from generated names
        n = name.lower()
        if n.startswith(("nvenc", "pair_gpu", "counter_nvenc")):
            return "nvenc"
        if n.startswith(("x264", "pair_cpu", "counter_x264", "rgb_probe",
                         "atem", "telemed")):
            return "libx264"
        return "other"

    def per_reader_pct_exact(reader: str, codec: str) -> float:
        sub = [r for r in rows
               if r["reader"] == reader and clip_codec(r["clip"]) == codec
               and r["clip"] != "rgb_probe"]
        if not sub:
            return float("nan")
        return 100.0 * sum(r["pixel_exact"] for r in sub) / len(sub)

    def per_reader_min_psnr_nonkey(reader: str, codec: str) -> float:
        sub = [r for r in rows
               if r["reader"] == reader and clip_codec(r["clip"]) == codec
               and not r["is_keyframe"]
               and not r["pixel_exact"]
               and r["clip"] != "rgb_probe"]
        if not sub:
            return float("inf")
        psnrs = [r["psnr_db"] for r in sub if not math.isnan(r["psnr_db"])]
        return min(psnrs) if psnrs else float("inf")

    pyav_x264 = per_reader_pct_exact("pyav_toc", "libx264")
    pyav_nvenc = per_reader_pct_exact("pyav_toc", "nvenc")
    pyav_min_psnr_x264 = per_reader_min_psnr_nonkey("pyav_toc", "libx264")
    pyav_min_psnr_nvenc = per_reader_min_psnr_nonkey("pyav_toc", "nvenc")

    decord_nvenc = per_reader_pct_exact("decord", "nvenc")

    # Seek correctness on counter clips
    seek_rows = [r for r in rows
                 if r["reader"] == "pyav_toc"
                 and r["clip"].startswith("counter_")]
    seek_total = len(seek_rows)
    seek_correct = sum(r.get("seek_correct", False) for r in seek_rows)
    seek_pct = (100.0 * seek_correct / seek_total) if seek_total else float("nan")

    # Decision rule
    ship_ok = True
    if not (pyav_x264 >= 100.0):
        ship_ok = False
        notes.append(f"pyav_toc libx264 pixel-exact = {pyav_x264:.1f}% (< 100%)")
    if not (pyav_nvenc >= 99.0):
        ship_ok = False
        notes.append(f"pyav_toc nvenc pixel-exact = {pyav_nvenc:.1f}% (< 99%)")
    if not (pyav_min_psnr_x264 >= 50.0):
        ship_ok = False
        notes.append(f"pyav_toc libx264 min PSNR (non-exact, non-key) = {pyav_min_psnr_x264:.1f} dB (< 50)")
    if not (pyav_min_psnr_nvenc >= 50.0):
        ship_ok = False
        notes.append(f"pyav_toc nvenc min PSNR (non-exact, non-key) = {pyav_min_psnr_nvenc:.1f} dB (< 50)")
    if not math.isnan(seek_pct) and seek_pct < 100.0:
        ship_ok = False
        notes.append(f"pyav_toc seek correctness on counter clips = {seek_pct:.1f}% (< 100%)")
    if workflow_array_match_pct is not None and workflow_array_match_pct < 100.0:
        ship_ok = False
        notes.append(f"workflow PNG ndarray match = {workflow_array_match_pct:.1f}% (< 100%)")
    if rgb_decord != rgb_pyav_toc:
        ship_ok = False
        notes.append(f"channel-order mismatch: decord={rgb_decord}, pyav_toc={rgb_pyav_toc}")

    # Triangulate the INCONCLUSIVE case: decord ALSO failing the oracle on NVENC
    if (not ship_ok) and (not math.isnan(decord_nvenc)) and decord_nvenc < 99.0:
        verdict = "INCONCLUSIVE_NEEDS_INVESTIGATION"
        notes.insert(0,
            f"both decord AND pyav_toc disagree with the oracle on NVENC "
            f"(decord_nvenc_exact={decord_nvenc:.1f}%, pyav_toc_nvenc_exact={pyav_nvenc:.1f}%). "
            "Possible interpretations: oracle is wrong, NVENC output is unstable, "
            "or the codebase has been wrong about NVENC all along.")
        return verdict, notes

    if ship_ok:
        # Strengthen if decord itself fails on NVENC
        if not math.isnan(decord_nvenc) and decord_nvenc < 99.0:
            notes.insert(0,
                f"decord fails oracle on NVENC ({decord_nvenc:.1f}% exact); "
                "pyav_toc passes — strong evidence for the swap.")
        return "SHIP_PYAV_TOC", notes

    return "KEEP_DECORD", notes


def write_report(out_path: Path,
                 rows: list[dict],
                 rgb_results: dict[str, str],
                 workflow_results: dict,
                 env_fingerprint: dict[str, str]) -> str:
    """Write the Markdown report; return the verdict string."""
    verdict, notes = derive_verdict(
        rows,
        workflow_results.get("array_match_pct"),
        rgb_results.get("decord", "?"),
        rgb_results.get("pyav_toc", "?"),
    )

    lines: list[str] = []
    lines.append(f"## VERDICT: {verdict}")
    lines.append("")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")

    # Section 2: RGB/BGR
    lines.append("## Channel order (RGB vs BGR)")
    lines.append("")
    lines.append("| reader | top-pixel dominant channel | verdict |")
    lines.append("|---|---|---|")
    for reader, verd in rgb_results.items():
        lines.append(f"| `{reader}` | — | **{verd}** |")
    lines.append("")

    # Section 3: per-clip x per-reader aggregates
    lines.append("## Aggregate parity (per clip × reader)")
    lines.append("")
    lines.append("| clip | reader | n | % exact | % matches_oracle | "
                 "% exact (key only) | min PSNR (non-key, non-exact) | median L1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    by = {}
    for r in rows:
        key = (r["clip"], r["reader"])
        by.setdefault(key, []).append(r)
    for (clip, reader), sub in sorted(by.items()):
        n = len(sub)
        pct_exact = 100.0 * sum(r["pixel_exact"] for r in sub) / n
        pct_match = 100.0 * sum(r["matches_oracle"] for r in sub) / n
        key_sub = [r for r in sub if r["is_keyframe"]]
        pct_exact_key = (100.0 * sum(r["pixel_exact"] for r in key_sub) / len(key_sub)
                         if key_sub else float("nan"))
        nk_sub = [r for r in sub if not r["is_keyframe"] and not r["pixel_exact"]]
        min_psnr = (min((r["psnr_db"] for r in nk_sub if not math.isnan(r["psnr_db"])),
                        default=float("inf"))
                    if nk_sub else float("inf"))
        med_l1 = float(np.median([r["l1_mean"] for r in sub]))
        lines.append(
            f"| `{clip}` | `{reader}` | {n} | {pct_exact:5.1f} | {pct_match:5.1f} | "
            f"{pct_exact_key if not math.isnan(pct_exact_key) else 'n/a':>5} | "
            f"{min_psnr:.1f} | {med_l1:.3f} |"
        )
    lines.append("")

    # Section 4: workflow-level
    lines.append("## Workflow-level parity (DUSTrack `_extract_frames` PNG diff)")
    lines.append("")
    for k, v in workflow_results.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    # Section 5: per-frame disagreements (filtered)
    lines.append("## Disagreements with oracle (psnr < 50 dB or shape mismatch)")
    lines.append("")
    bad = [r for r in rows if not r["matches_oracle"]]
    if not bad:
        lines.append("_(none)_")
    else:
        lines.append("| clip | reader | index | is_key | shape match | l1 | psnr_db |")
        lines.append("|---|---|---:|---|---|---:|---:|")
        for r in sorted(bad, key=lambda r: (r["clip"], r["reader"], r["index"])):
            shape_ok = "yes" if not math.isnan(r["psnr_db"]) else "NO"
            lines.append(
                f"| `{r['clip']}` | `{r['reader']}` | {r['index']} | "
                f"{'key' if r['is_keyframe'] else '-'} | {shape_ok} | "
                f"{r['l1_mean']:.3f} | {r['psnr_db']:.1f} |"
            )
    lines.append("")

    # Section 6: env fingerprint
    lines.append("## Environment fingerprint")
    lines.append("")
    for k, v in env_fingerprint.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return verdict
