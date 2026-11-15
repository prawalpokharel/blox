"""Tests for placement/criticality_placement.py. Run with `python -m pytest tests`."""

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from placement.criticality_placement import (  # noqa: E402
    DependencyCriticalityPlacement,
    _ClusterView,
)


def make_gpu_df(n_nodes=4, gpus_per_node=4, busy=None):
    """busy: {gpu_id: job_id}"""
    busy = busy or {}
    rows = []
    for node in range(n_nodes):
        for local in range(gpus_per_node):
            gid = node * gpus_per_node + local
            rows.append(
                {
                    "GPU_ID": gid,
                    "GPU_UUID": str(local),
                    "Local_GPU_ID": local,
                    "Node_ID": node,
                    "IP_addr": "",
                    "IN_USE": gid in busy,
                    "JOB_IDS": busy.get(gid),
                }
            )
    return pd.DataFrame(rows)


def make_job(num_gpus, running=False):
    return {"num_GPUs": num_gpus, "is_running": running, "job_priority": 999}


def run_place(policy, jobs, order, gpu_df):
    job_state = SimpleNamespace(active_jobs=jobs)
    cluster_state = SimpleNamespace(server_map={}, gpu_df=gpu_df)
    schedule = {"job_order": [(j, jobs[j]) for j in order], "run_all_jobs": False}
    return policy.place(job_state, cluster_state, schedule)


def gpu_nodes(gpus, gpus_per_node=4):
    return {g // gpus_per_node for g in gpus}


def test_single_gpu_job_avoids_critical_node():
    # Nodes 0 and 1 each hold one busy GPU, nodes 2 and 3 are full. The new
    # job must pick node 0 or 1, and node 0 is declared critical.
    busy = {0: 10, 4: 11}
    busy.update({g: 12 for g in range(8, 12)})
    busy.update({g: 13 for g in range(12, 16)})
    jobs = {
        10: make_job(1, True), 11: make_job(1, True),
        12: make_job(4, True), 13: make_job(4, True), 20: make_job(1),
    }
    policy = DependencyCriticalityPlacement(critical_nodes=[0], rebalance=False)
    _, launch = run_place(policy, jobs, [10, 11, 12, 13, 20], make_gpu_df(busy=busy))
    assert gpu_nodes(launch[20]) == {1}


def test_job_that_fits_one_node_is_not_split():
    jobs = {1: make_job(4)}
    policy = DependencyCriticalityPlacement(rebalance=False)
    _, launch = run_place(policy, jobs, [1], make_gpu_df())
    assert len(launch[1]) == 4
    assert len(gpu_nodes(launch[1])) == 1


def test_large_job_uses_fewest_nodes():
    jobs = {1: make_job(6)}
    policy = DependencyCriticalityPlacement(rebalance=False)
    _, launch = run_place(policy, jobs, [1], make_gpu_df())
    assert len(launch[1]) == 6
    assert len(gpu_nodes(launch[1])) == 2


def test_large_job_steers_off_critical_nodes():
    jobs = {1: make_job(8)}
    policy = DependencyCriticalityPlacement(critical_nodes=[0, 1], rebalance=False)
    _, launch = run_place(policy, jobs, [1], make_gpu_df())
    assert gpu_nodes(launch[1]) == {2, 3}


def test_preempts_lowest_priority_when_full():
    busy = {g: 1 for g in range(0, 8)}
    busy.update({g: 2 for g in range(8, 16)})
    jobs = {1: make_job(8, True), 2: make_job(8, True), 3: make_job(4)}
    policy = DependencyCriticalityPlacement(rebalance=False)
    # Job 3 has top priority, job 2 the lowest.
    terminate, launch = run_place(policy, jobs, [3, 1, 2], make_gpu_df(busy=busy))
    assert terminate == [2]
    assert set(launch) == {3}
    assert gpu_nodes(launch[3]) <= {2, 3}


def test_stops_at_first_job_that_cannot_fit():
    busy = {g: 1 for g in range(0, 16)}
    jobs = {1: make_job(16, True), 2: make_job(4), 3: make_job(1)}
    policy = DependencyCriticalityPlacement(rebalance=False)
    # Job 1 is higher priority than 2 and 3, so nothing is preempted.
    terminate, launch = run_place(policy, jobs, [1, 2, 3], make_gpu_df(busy=busy))
    assert terminate == [] and launch == {}


def test_objective_matches_iemcon_formula():
    policy = DependencyCriticalityPlacement(lam=0.35, g=0.5)
    used = np.array([4.0, 2.0, 1.0, 0.0])
    cap = np.full(4, 4.0)
    crit = np.array([1.0, 0.2, 0.2, 0.2])
    p = used / used.sum()
    q = p[p > 0]
    h = -(q * np.log(q)).sum() / math.log(4)
    phi = (crit * p).sum() / crit.max()
    u = used / cap
    power = np.where(u > 0, 550 + 950 * u ** 1.2, 0.0)
    e = power.sum() / (4 * 1500)
    expected = 0.65 * (h - 0.5 * phi) - 0.35 * e
    assert policy.objective(used, cap, crit) == pytest.approx(expected, rel=1e-12)


def test_adaptive_lambda_follows_hysteresis():
    policy = DependencyCriticalityPlacement(adaptive=True, lam=0.35)
    policy.temp = {0: 0.9, 1: 0.1}
    policy._recalibrate()
    assert policy.lam == pytest.approx(0.35 - 6 * 0.02)
    policy.temp = {0: 0.1, 1: 0.1}
    policy._recalibrate()
    assert policy.lam == pytest.approx(0.35 - 6 * 0.02 + 0.02)
    policy.temp = {0: 0.78 * 0.95}
    before = policy.lam
    policy._recalibrate()
    assert policy.lam == pytest.approx(before)


def test_rebalance_moves_small_job_off_hot_node():
    busy = {0: 5, 1: 6, 2: 6, 3: 6}
    jobs = {5: make_job(1, True), 6: make_job(3, True)}
    policy = DependencyCriticalityPlacement(rebalance=True, w_m=0.0)
    policy.temp = {0: 0.95, 1: 0.0, 2: 0.0, 3: 0.0}
    policy.alpha_t = 1.0  # keep the preset temperatures for this round
    terminate, launch = run_place(policy, jobs, [5, 6], make_gpu_df(busy=busy))
    assert terminate == [5]
    assert gpu_nodes(launch[5]) & {1, 2, 3}
    assert policy.moves_last_round == 1


def test_place_does_not_change_blox_state():
    df = make_gpu_df(busy={0: 1})
    jobs = {1: make_job(1, True), 2: make_job(2)}
    snapshot_df = df.copy()
    run_place(DependencyCriticalityPlacement(), jobs, [1, 2], df)
    pd.testing.assert_frame_equal(df, snapshot_df)
    assert jobs[2]["is_running"] is False


def test_span_graph_marks_shared_node_as_most_central():
    # Job 7 spans nodes 0 and 1, job 8 spans nodes 1 and 2.
    busy = {0: 7, 4: 7, 5: 8, 8: 8}
    view = _ClusterView(make_gpu_df(busy=busy))
    graph = view.span_graph()
    assert set(graph.edges()) == {(0, 1), (1, 2)}
    crit = DependencyCriticalityPlacement()._criticality(view)
    assert crit[1] == pytest.approx(1.0)
    assert crit[1] > crit[0] > crit[3]
    assert crit[3] == pytest.approx(0.2)


def test_gavel_schedule_is_rejected():
    jobs = {1: make_job(1)}
    policy = DependencyCriticalityPlacement()
    schedule = {"job_order": [((1, "v100"), jobs[1])], "scheduler": "Gavel"}
    with pytest.raises(NotImplementedError):
        policy.place(
            SimpleNamespace(active_jobs=jobs),
            SimpleNamespace(server_map={}, gpu_df=make_gpu_df()),
            schedule,
        )


def test_governance_floor_keeps_jobs_off_critical_nodes_until_full():
    # Node 0 is critical. With the floor on, jobs avoid it while the others have room,
    # even where the objective alone would pick it.
    jobs = {1: make_job(4), 2: make_job(4), 3: make_job(4), 4: make_job(4)}
    policy = DependencyCriticalityPlacement(critical_nodes=[0], governance_floor=True, rebalance=False)
    _, launch = run_place(policy, jobs, [1, 2, 3, 4], make_gpu_df())
    nodes = [gpu_nodes(launch[j]) for j in (1, 2, 3)]
    assert {0} not in nodes
    assert gpu_nodes(launch[4]) == {0}


def test_governance_floor_off_matches_default_choice():
    jobs = {1: make_job(2)}
    a = DependencyCriticalityPlacement(critical_nodes=[0], rebalance=False)
    b = DependencyCriticalityPlacement(critical_nodes=[0], governance_floor=False, rebalance=False)
    assert run_place(a, jobs, [1], make_gpu_df()) == run_place(b, jobs, [1], make_gpu_df())


def test_near_ties_break_to_the_lowest_index():
    policy = DependencyCriticalityPlacement()
    scores = [0.5, 0.5 + 1e-15, 0.4]
    assert policy._argmax(scores) == 0
    assert DependencyCriticalityPlacement(tie_tol=0.0)._argmax(scores) == 1
    assert policy._argmax([0.1, 0.3, 0.2]) == 1
