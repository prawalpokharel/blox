"""Dependency-criticality placement for Blox.

This policy ports the placement rule of the Centrality-Entropy Index (CEI)
scheduler from P. Pokharel, "Self-Tuning Governance of the Energy-Resilience
Tradeoff in GPU Cluster Scheduling", IEMCON 2026, to Blox's placement
interface.

Every candidate placement is scored with the IEMCON objective

    J = (1 - lam) * (H(p) / log(n) - g * Phi) - lam * E

    p    share of busy GPUs on each of the n nodes
    H    Shannon entropy of p (spreading load raises it)
    Phi  sum_i c_i * p_i / max(c), the load sitting on critical nodes
    E    cluster power under the IEMCON node model, divided by n * P_peak

Node criticality c_i is c_low + (c_high - c_low) * max(tier_i, span_i):

    tier_i  1 for nodes an operator declares critical (``critical_nodes``),
            the high-centrality tier of the IEMCON study
    span_i  the node's centrality in the graph whose edges join nodes that
            share a running multi-node job, scaled to [0, 1]. Blox has no
            network topology, so this graph is the dependency structure it
            does have: a node that many distributed jobs pass through takes
            several jobs down with it when it fails.

A job that fits on one node goes to the feasible node with the best J. A larger
job takes the fewest nodes possible, and J chooses which ones. Nodes whose
thermal proxy is near the soft limit are avoided when any other node fits. When
nothing fits, lower-priority running jobs are preempted in reverse priority
order, as in Blox's default placement.

After placing, up to two single-node jobs move off nodes above the soft
thermal limit when the move raises J by more than the migration cost
``w_m``. With ``adaptive=True``, lam follows the IEMCON hysteresis band:
it drops by 6 * eta after a soft-limit violation, holds near the limit, and
rises by eta when the cluster is cool.

The thermal proxy is the IEMCON first-order model
T_i <- a * T_i + (1 - a) * 1.08 * u_i, with u_i the node's busy-GPU share.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .placement_policy import JobPlacement

__all__ = ["DependencyCriticalityPlacement"]


class DependencyCriticalityPlacement(JobPlacement):
    """Criticality-aware placement (CEI objective) for Blox."""

    name = "DependencyCriticality"

    def __init__(
        self,
        args=None,
        critical_nodes: Optional[Sequence[int]] = None,
        lam: float = 0.35,
        g: float = 0.5,
        adaptive: bool = False,
        use_span_graph: bool = True,
        rebalance: bool = True,
        w_m: float = 0.02,
        eta: float = 0.02,
        lam_bounds: Tuple[float, float] = (0.15, 0.90),
        c_low: float = 0.2,
        c_high: float = 1.0,
        soft_t: float = 0.78,
        alpha_t: float = 0.7,
        p_idle: float = 550.0,
        p_peak: float = 1500.0,
        k: float = 1.2,
    ):
        if critical_nodes is None and args is not None:
            critical_nodes = getattr(args, "critical_nodes", None)
        self.critical_nodes = set(int(n) for n in (critical_nodes or []))
        self.lam = float(lam)
        self.g = float(g)
        self.adaptive = bool(adaptive)
        self.use_span_graph = bool(use_span_graph)
        self.rebalance = bool(rebalance)
        self.w_m = float(w_m)
        self.eta = float(eta)
        self.lam_lo, self.lam_hi = lam_bounds
        self.c_low, self.c_high = float(c_low), float(c_high)
        self.soft_t = float(soft_t)
        self.alpha_t = float(alpha_t)
        self.p_idle, self.p_peak, self.k = float(p_idle), float(p_peak), float(k)
        self.temp: Dict[int, float] = {}
        self.moves_last_round = 0

    # Blox calls place(job_state, cluster_state, new_job_schedule). The
    # decorator hands the policy deep copies, so nothing here changes Blox's
    # own state.
    @JobPlacement.copy_arguments
    def place(
        self,
        active_jobs: dict,
        new_job_schedule: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        **kwargs,
    ) -> Tuple[List[int], Dict[int, List[int]]]:
        if new_job_schedule.get("scheduler") is not None:
            raise NotImplementedError(
                "DependencyCriticalityPlacement supports schedulers that return "
                "a plain job order (FIFO, LAS, SRTF, Tiresias)."
            )
        job_order = new_job_schedule["job_order"]
        view = _ClusterView(gpu_df)
        self._update_temperature(view)
        crit = self._criticality(view)

        to_terminate: List[int] = []
        to_launch: Dict[int, List[int]] = {}
        running_at_start = {
            jid for jid, job in active_jobs.items() if job.get("is_running")
        }

        for idx, (jid, _) in enumerate(job_order):
            job = active_jobs[jid]
            if job["is_running"]:
                continue
            gpus = self._choose(int(job["num_GPUs"]), view, crit)
            if gpus is None:
                for rev_idx in range(1, len(job_order) - idx):
                    victim = job_order[-rev_idx][0]
                    if victim in running_at_start and active_jobs[victim]["is_running"]:
                        to_terminate.append(victim)
                        active_jobs[victim]["is_running"] = False
                        view.release(victim)
                        gpus = self._choose(int(job["num_GPUs"]), view, crit)
                        if gpus is not None:
                            break
            if gpus is None:
                break
            to_launch[jid] = gpus
            view.occupy(gpus, jid)

        self.moves_last_round = 0
        if self.rebalance:
            for jid, gpus in self._rebalance_moves(
                active_jobs, view, crit, set(to_launch), set(to_terminate)
            ):
                to_terminate.append(jid)
                to_launch[jid] = gpus
                self.moves_last_round += 1

        if self.adaptive:
            self._recalibrate()
        return (to_terminate, to_launch)

    # scoring --------------------------------------------------------------
    def objective(self, used: np.ndarray, cap: np.ndarray, crit: np.ndarray) -> float:
        """IEMCON objective J for one vector of busy GPUs per node."""
        return float(self._objective_batch(used[None, :], cap, crit)[0])

    def _objective_batch(self, used: np.ndarray, cap: np.ndarray, crit: np.ndarray) -> np.ndarray:
        n = used.shape[1]
        tot = used.sum(axis=1, keepdims=True)
        p = np.divide(used, tot, out=np.zeros_like(used, dtype=float), where=tot > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            plogp = np.where(p > 0, p * np.log(p), 0.0)
        h = -plogp.sum(axis=1) / math.log(n) if n > 1 else np.zeros(len(used))
        phi = (p * crit[None, :]).sum(axis=1) / crit.max()
        u = np.clip(used / cap[None, :], 0.0, 1.0)
        power = np.where(u > 1e-9, self.p_idle + (self.p_peak - self.p_idle) * u ** self.k, 0.0)
        energy = power.sum(axis=1) / (n * self.p_peak)
        return (1.0 - self.lam) * (h - self.g * phi) - self.lam * energy

    def _criticality(self, view: "_ClusterView") -> np.ndarray:
        tier = np.array([1.0 if n in self.critical_nodes else 0.0 for n in view.nodes])
        span = np.zeros(len(view.nodes))
        if self.use_span_graph:
            graph = view.span_graph()
            if graph.number_of_edges() > 0:
                score = _composite_centrality(graph)
                for pos, node in enumerate(view.nodes):
                    span[pos] = score.get(node, 0.0)
        return self.c_low + (self.c_high - self.c_low) * np.maximum(tier, span)

    # choosing GPUs --------------------------------------------------------
    def _choose(self, demand: int, view: "_ClusterView", crit: np.ndarray) -> Optional[List[int]]:
        free = view.free_count()
        if free.sum() < demand:
            return None
        temp = np.array([self.temp.get(n, 0.0) for n in view.nodes])
        cool = temp < self.soft_t * 0.95

        single = np.where(free >= demand)[0]
        if len(single) > 0:
            pick = single[cool[single]] if cool[single].any() else single
            trial = np.repeat(view.used[None, :], len(pick), axis=0).astype(float)
            trial[np.arange(len(pick)), pick] += demand
            best = pick[int(np.argmax(self._objective_batch(trial, view.cap, crit)))]
            return view.free_gpus(best)[:demand]

        # Spread over the fewest nodes possible, letting J choose which.
        used = view.used.astype(float).copy()
        left = free.copy()
        remaining = demand
        chosen: List[int] = []
        while remaining > 0:
            cand = np.where(left > 0)[0]
            if len(cand) == 0:
                return None
            take = np.minimum(left[cand], remaining)
            cand = cand[take == take.max()]
            if cool[cand].any():
                cand = cand[cool[cand]]
            t = int(min(left[cand[0]], remaining))
            trial = np.repeat(used[None, :], len(cand), axis=0)
            trial[np.arange(len(cand)), cand] += t
            best = cand[int(np.argmax(self._objective_batch(trial, view.cap, crit)))]
            chosen.extend(view.free_gpus(best)[:t])
            used[best] += t
            left[best] -= t
            remaining -= t
        return chosen

    def _rebalance_moves(self, active_jobs, view, crit, launched, terminated):
        moves = []
        base = self.objective(view.used.astype(float), view.cap, crit)
        temp = np.array([self.temp.get(n, 0.0) for n in view.nodes])
        for _ in range(2):
            hot = np.where(temp > self.soft_t)[0]
            if len(hot) == 0:
                break
            h = hot[int(np.argmax(temp[hot]))]
            victims = [
                jid
                for jid in view.jobs_on_node(h)
                if jid not in launched and jid not in terminated
                and active_jobs.get(jid, {}).get("is_running")
                and len(view.job_nodes(jid)) == 1
            ]
            if not victims:
                break
            victims.sort(key=lambda j: (len(view.job_gpus(j)), j))
            jid = victims[0]
            size = len(view.job_gpus(jid))
            free = view.free_count()
            targets = [
                i for i in range(len(view.nodes))
                if i != h and free[i] >= size and temp[i] < self.soft_t * 0.9
            ]
            if not targets:
                break
            trial = np.repeat(view.used[None, :].astype(float), len(targets), axis=0)
            trial[:, h] -= size
            trial[np.arange(len(targets)), targets] += size
            scores = self._objective_batch(trial, view.cap, crit)
            best = int(np.argmax(scores))
            if scores[best] - base - self.w_m <= 0:
                break
            dst = targets[best]
            view.release(jid)
            gpus = view.free_gpus(dst)[:size]
            view.occupy(gpus, jid)
            moves.append((jid, gpus))
            launched.add(jid)
            base = float(scores[best])
        return moves

    # state updated once per round -----------------------------------------
    def _update_temperature(self, view: "_ClusterView") -> None:
        util = view.used / view.cap
        for pos, node in enumerate(view.nodes):
            prev = self.temp.get(node, 0.0)
            self.temp[node] = self.alpha_t * prev + (1.0 - self.alpha_t) * 1.08 * float(util[pos])

    def _recalibrate(self) -> None:
        temps = np.array(list(self.temp.values())) if self.temp else np.zeros(1)
        if (temps > self.soft_t).any():
            step = -6.0 * self.eta
        elif (temps > self.soft_t * 0.92).any():
            step = 0.0
        else:
            step = self.eta
        self.lam = float(np.clip(self.lam + step, self.lam_lo, self.lam_hi))


class _ClusterView:
    """Numpy view of Blox's gpu_df, updated as the policy places jobs."""

    def __init__(self, gpu_df: pd.DataFrame):
        df = gpu_df.sort_values("GPU_ID")
        self.gpu_ids = df["GPU_ID"].astype(int).to_numpy()
        node_col = df["Node_ID"].astype(int).to_numpy()
        self.nodes = sorted(set(node_col.tolist()))
        pos = {n: i for i, n in enumerate(self.nodes)}
        self.gpu_node = np.array([pos[n] for n in node_col], dtype=int)
        self.cap = np.bincount(self.gpu_node, minlength=len(self.nodes)).astype(float)
        self.gpu_job = [
            (None if (j is None or (isinstance(j, float) and math.isnan(j))) else int(j))
            if bool(busy) else None
            for j, busy in zip(df["JOB_IDS"].tolist(), df["IN_USE"].tolist())
        ]
        self._refresh()

    def _refresh(self) -> None:
        busy = np.array([j is not None for j in self.gpu_job])
        self.used = np.bincount(self.gpu_node[busy], minlength=len(self.nodes)).astype(float)

    def free_count(self) -> np.ndarray:
        return self.cap - self.used

    def free_gpus(self, node_pos: int) -> List[int]:
        mask = (self.gpu_node == node_pos) & np.array([j is None for j in self.gpu_job])
        return self.gpu_ids[mask].tolist()

    def occupy(self, gpus: List[int], jid: int) -> None:
        index = {g: i for i, g in enumerate(self.gpu_ids.tolist())}
        for gid in gpus:
            self.gpu_job[index[gid]] = int(jid)
        self._refresh()

    def release(self, jid: int) -> None:
        self.gpu_job = [None if j == jid else j for j in self.gpu_job]
        self._refresh()

    def job_gpus(self, jid: int) -> List[int]:
        return [g for g, j in zip(self.gpu_ids.tolist(), self.gpu_job) if j == jid]

    def job_nodes(self, jid: int) -> set:
        return {int(n) for n, j in zip(self.gpu_node, self.gpu_job) if j == jid}

    def jobs_on_node(self, node_pos: int) -> List[int]:
        return sorted({j for n, j in zip(self.gpu_node, self.gpu_job) if n == node_pos and j is not None})

    def span_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        by_job: Dict[int, set] = {}
        for n, j in zip(self.gpu_node, self.gpu_job):
            if j is not None:
                by_job.setdefault(j, set()).add(self.nodes[n])
        for members in by_job.values():
            members = sorted(members)
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    graph.add_edge(members[a], members[b])
        return graph


def _composite_centrality(graph: nx.Graph) -> Dict[int, float]:
    """0.35 PageRank + 0.30 betweenness + 0.20 degree + 0.15 closeness, scaled to [0, 1].

    Same weights as the CEI calculator. Isolated nodes score 0.
    """
    active = graph.subgraph([n for n in graph if graph.degree(n) > 0])
    if active.number_of_nodes() == 0:
        return {}
    pr = nx.pagerank(active, alpha=0.85)
    bw = nx.betweenness_centrality(active)
    dg = nx.degree_centrality(active)
    cl = nx.closeness_centrality(active)
    comp = {n: 0.35 * pr[n] + 0.30 * bw[n] + 0.20 * dg[n] + 0.15 * cl[n] for n in active}
    top = max(comp.values())
    return {n: (v / top if top > 0 else 0.0) for n, v in comp.items()}
