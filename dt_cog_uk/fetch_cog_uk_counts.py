#!/usr/bin/env python3
"""Count COG-UK SARS-CoV-2 genomes sampled in ENGLAND per week, by constellation group.

WHY THIS EXISTS
---------------
The ONE source of variant proportions in this repo (since 2026-10-22; the ONS SGTF and UKHSA lineage
caches, `dt_ons_variant/` and `dt_ukhsa/`, were removed). COG-UK's public metadata has one row per UK
genome, with its nation (`adm1`), `sample_date` and lineage, so the weekly England genome counts give
both the variant shares and their sample size. The `VariantNGM` renewal (`src/variant_data.jl`) reads
the counts; 1j draws the shares and writes them to `cog_uk_england_weekly_share.csv` beside this file.

SOURCE
------
  https://cog-uk.s3.climb.ac.uk/phylogenetics/latest/cog_metadata.csv.gz   (~165 MB, all UK)
Downloaded to a temporary directory and NOT kept; only the weekly aggregate is committed. Python
because Project.toml carries no HTTP package (adding one would force an HPC image rebuild).

WHAT IT WRITES (beside this file, committed)
--------------------------------------------
  cog_uk_england_weekly.csv   week_start, group, n_genomes, n_pillar2
  `week_start` is the MONDAY of the sample-date week, matching the framework's Monday weeks.
  `group` comes from `scorpio_call` (the constellation call), NOT from parsing Pango names — Pango
  aliases (BA.1 → BC/BD…, Delta → AY.*) make prefix rules wrong:
      Alpha         scorpio "Alpha (…)" or "B.1.1.7-like…"
      Delta         scorpio "Delta (…)"
      Omicron BA.1  scorpio "Omicron (BA.1-like)"
      Omicron BA.2  scorpio "Omicron (BA.2-like)"
      Omicron other any other "Omicron …" / "Probable Omicron …" call
      Other         no call, or any other constellation (Beta, Gamma, Eta, …)
  A failed or low-quality sequence has no genome, so it is absent from every group here.

SELF-TESTS (every invocation)
-----------------------------
  1. every row England; week_start a Monday; n_genomes > 0; n_pillar2 <= n_genomes
  2. weeks contiguous from 2020-11-16 to 2022-02-21 (the 1j span)
  (A third, REPORTED test compared these shares with UKHSA's lineage %; it went with `dt_ukhsa/`.
  It had measured a median 0.00–0.18 pp difference for Alpha/Delta/BA.1/BA.2 over 2021-01-04 …
  2022-02-21.)

Usage:  python3 -I dt_cog_uk/fetch_cog_uk_counts.py [--refresh] [--metadata PATH]
        (`--metadata` reads an already-downloaded cog_metadata.csv.gz instead of fetching; dev only)
"""

import argparse
import os
import sys
import tempfile
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
URL = "https://cog-uk.s3.climb.ac.uk/phylogenetics/latest/cog_metadata.csv.gz"
OUT_CSV = os.path.join(HERE, "cog_uk_england_weekly.csv")
SPAN = (pd.Timestamp("2020-11-16"), pd.Timestamp("2022-02-21"))   # Monday weeks covering 1j's span
GROUPS = ["Alpha", "Delta", "Omicron BA.1", "Omicron BA.2", "Omicron other", "Other"]


def group_of(call):
    if not isinstance(call, str) or not call:
        return "Other"
    if call.startswith("Alpha") or call.startswith("B.1.1.7-like"):
        return "Alpha"
    if call.startswith("Delta"):
        return "Delta"
    if call == "Omicron (BA.1-like)":
        return "Omicron BA.1"
    if call == "Omicron (BA.2-like)":
        return "Omicron BA.2"
    if "Omicron" in call:
        return "Omicron other"
    return "Other"


def download(path, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            return
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(10 * 2 ** k)


def build(metadata):
    parts = []
    for ch in pd.read_csv(metadata, usecols=["adm1", "is_pillar_2", "sample_date", "scorpio_call"],
                          dtype=str, chunksize=1_000_000):
        ch = ch[ch.adm1 == "UK-ENG"]
        parts.append(ch.groupby(["sample_date", "is_pillar_2", "scorpio_call"], dropna=False)
                     .size().rename("n").reset_index())
    d = pd.concat(parts)
    d["date"] = pd.to_datetime(d.sample_date, errors="coerce")
    bad = int(d.loc[d.date.isna(), "n"].sum())
    print(f"    {int(d.n.sum())} England genomes; {bad} with an unparseable sample_date dropped")
    d = d.dropna(subset=["date"])
    d["week_start"] = d.date - pd.to_timedelta(d.date.dt.dayofweek, unit="D")
    d["group"] = d.scorpio_call.map(group_of)
    d["p2"] = d.n.where(d.is_pillar_2 == "Y", 0)
    out = (d.groupby(["week_start", "group"], as_index=False)
           .agg(n_genomes=("n", "sum"), n_pillar2=("p2", "sum")))
    return out.sort_values(["week_start", "group"]).reset_index(drop=True)


def self_test(df):
    assert (df.week_start.dt.dayofweek == 0).all(), "a week_start is not a Monday"
    assert (df.n_genomes > 0).all() and (df.n_pillar2 <= df.n_genomes).all()
    assert set(df.group) <= set(GROUPS), f"unexpected groups {set(df.group) - set(GROUPS)}"
    wk = sorted(df.week_start.unique())
    span = pd.date_range(*SPAN, freq="7D")
    missing = sorted(set(span) - set(wk))
    assert not missing, f"weeks with no England genome in the 1j span: {[m.date() for m in missing]}"
    tot = df[df.week_start.between(*SPAN)].groupby("week_start").n_genomes.sum()
    print(f"    [1-2] {len(wk)} weeks {wk[0].date()} .. {wk[-1].date()}; 1j span contiguous, "
          f"{int(tot.min())}-{int(tot.max())} genomes/week")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="rebuild even when the cache exists")
    ap.add_argument("--metadata", help="use this cog_metadata.csv.gz instead of downloading")
    args = ap.parse_args()
    have = os.path.exists(OUT_CSV)
    if have and not args.refresh:
        print(f"dt_cog_uk: {os.path.basename(OUT_CSV)} exists; self-testing the cache (--refresh to rebuild)")
        df = pd.read_csv(OUT_CSV, parse_dates=["week_start"])
    elif args.metadata:
        print(f"dt_cog_uk: aggregating {args.metadata}")
        df = build(args.metadata)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cog_metadata.csv.gz")
            print(f"dt_cog_uk: downloading {URL} (~165 MB) ...")
            download(path)
            df = build(path)
    print("  self-tests:")
    self_test(df)
    if not have or args.refresh:
        df.to_csv(OUT_CSV, index=False, date_format="%Y-%m-%d")
        print(f"  wrote {os.path.relpath(OUT_CSV, REPO)} ({len(df)} rows)")
    print("dt_cog_uk: OK")


if __name__ == "__main__":
    main()
