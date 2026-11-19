#!/usr/bin/env python3
"""Offline re-grade of banking_knowledge trajectories under the current tau2 grader.

banking_knowledge rewards on DB state (reward_basis = ["DB"]), so an ENV-only
evaluation reproduces the reward without any LLM grader calls. Retrieval tools
are irrelevant to replay (read-only and unknown tools are skipped), so the
environment is built with the ``no_knowledge`` variant: no API keys needed.

  python scripts/nexus/regrade_db.py IN.json OUT.json [--fresh-tasks]
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TAU2_DEFAULT_RETRIEVAL_VARIANT", "no_knowledge")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--fresh-tasks", action="store_true")
    a = ap.parse_args()

    from tau2.data_model.simulation import Results
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.scripts.evaluate_trajectories import compute_simulation_rewards

    results = Results.load(a.src)
    regraded = compute_simulation_rewards(results, evaluation_type=EvaluationType.ENV, fresh_tasks=a.fresh_tasks)
    changed = sum(
        (o.reward_info.reward if o.reward_info else 0) != (n.reward_info.reward if n.reward_info else 0)
        for o, n in zip(results.simulations, regraded.simulations)
    )
    regraded.save(a.dst)
    print(f"regraded {len(regraded.simulations)} sims, {changed} rewards changed -> {a.dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
