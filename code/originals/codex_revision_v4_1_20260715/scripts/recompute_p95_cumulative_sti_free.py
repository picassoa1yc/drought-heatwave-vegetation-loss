from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import recompute_sti_free as core


P95_EVENT_LEVEL_PATH = (
    core.ROOT
    / "output_df"
    / "global_v05_95p_robustness"
    / "event_analysis"
    / "event_multivariate_response_event_level.csv"
)


def assert_aligned(cumulative: pd.DataFrame, p95_meta: pd.DataFrame) -> dict[str, object]:
    if len(cumulative) != len(p95_meta):
        raise ValueError(
            f"P90 cumulative table ({len(cumulative)}) and P95 event table "
            f"({len(p95_meta)}) differ in length."
        )

    checks = {
        "land_cover_broad": cumulative["land_cover_broad"].astype(str).reset_index(drop=True).equals(
            p95_meta["land_cover_broad"].astype(str).reset_index(drop=True)
        ),
        "start_date": cumulative["start_date"].astype(str).reset_index(drop=True).equals(
            p95_meta["start_date"].astype(str).reset_index(drop=True)
        ),
        "duration": bool(
            np.allclose(
                pd.to_numeric(cumulative["duration"], errors="coerce"),
                pd.to_numeric(p95_meta["duration"], errors="coerce"),
                equal_nan=True,
            )
        ),
        "LOSStot": bool(
            np.allclose(
                pd.to_numeric(cumulative["LOSStot"], errors="coerce"),
                pd.to_numeric(p95_meta["LOSStot"], errors="coerce"),
                equal_nan=True,
                atol=1e-10,
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"P95 cumulative-event alignment failed for: {', '.join(failed)}")
    return checks


def main() -> None:
    print("Loading cumulative anomaly losses from the common NDVI-defined event record...")
    cumulative = pd.read_csv(core.CUMULATIVE_PATH)
    p95_meta = pd.read_csv(
        P95_EVENT_LEVEL_PATH,
        usecols=["event_type", "land_cover_broad", "duration", "start_date", "LOSStot"],
    )
    checks = assert_aligned(cumulative, p95_meta)

    original_event_type = cumulative["event_type"].astype(str).copy()
    cumulative["event_type"] = p95_meta["event_type"].astype(str).to_numpy()
    n_reclassified = int((original_event_type != cumulative["event_type"].astype(str)).sum())

    p95 = core.prepare_analysis_frame(cumulative)
    print(f"P95 compound/drought eligible events: {len(p95):,}")
    matches = core.build_matches(p95, core.CUM_MATCH_FEATURES, k_neighbors=3)
    paired = core.build_paired_table(p95, matches, core.CUM_OUTCOMES)
    summary = core.summarize_paired(paired, core.CUM_OUTCOMES, seed_offset=250)

    p90_summary = pd.read_csv(core.OUTPUT_DIR / "cumulative_global_att_sti_free.csv")
    reference_sd = p90_summary.set_index("outcome")["pooled_sd"].to_dict()
    summary = core.attach_fixed_standardization(summary, reference_sd)
    balance = core.compute_balance(p95, matches, core.CUM_MATCH_FEATURES)

    matches.to_csv(core.OUTPUT_DIR / "p95_cumulative_matches_sti_free.csv", index=False)
    paired.to_csv(core.OUTPUT_DIR / "p95_cumulative_paired_sti_free.csv", index=False)
    summary.to_csv(core.OUTPUT_DIR / "p95_cumulative_global_att_sti_free.csv", index=False)
    balance.to_csv(core.OUTPUT_DIR / "p95_cumulative_balance_sti_free.csv", index=False)

    threshold = p90_summary.merge(
        summary,
        on=["outcome", "outcome_label"],
        suffixes=("_p90", "_p95"),
        validate="one_to_one",
    )
    threshold.to_csv(
        core.OUTPUT_DIR / "cumulative_heat_threshold_p90_p95_sti_free.csv",
        index=False,
    )

    audit = {
        "common_event_rows": int(len(cumulative)),
        "alignment_checks": checks,
        "events_reclassified_between_p90_and_p95": n_reclassified,
        "p95_compound_events": int((cumulative["event_type"] == core.TREATED).sum()),
        "p95_drought_only_events": int((cumulative["event_type"] == core.CONTROL).sum()),
        "p95_matched_compound_events": int(len(matches)),
        "matching_features": core.CUM_MATCH_FEATURES,
        "maximum_absolute_post_match_smd": float(balance["smd_after"].abs().max()),
    }
    (core.OUTPUT_DIR / "p95_cumulative_reclassification_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print(summary[["outcome", "additional_effect", "additional_relative_pct", "n_treated"]].to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
