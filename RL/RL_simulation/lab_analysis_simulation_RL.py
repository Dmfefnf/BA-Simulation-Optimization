"""RL-ready lab-analysis simulation with explicit queues and active stations.

This module is separate from the original simulation and the BO version. The
first RL prototype controls only the Sorting station. The agent chooses an
interpretable dispatching rule, not a concrete order id.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import salabim as sim

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from RL_agents import BaselineAgent, QLearningAgent, RandomAgent, BaseAgent

HOURLY_RATES_PATH = BASE_DIR / "hourly_arrival_rates.csv"

MINUTE = 1
HOUR = 60 * MINUTE
DAY = 24 * HOUR

ANIMATION_SPEED = 2
CONST_ARRIVAL_RATE = 500 / DAY
DUE_RISK_THRESHOLD = 2 * HOUR
DEFAULT_RISK_T1 = 0
DEFAULT_RISK_WINDOW = 2 * HOUR
DUE_DATE_LOWER_BOUND = 4
DUE_DATE_UPPER_BOUND = 8
RATE_MULTIPLIER = 0.9

ACTION_NAMES = {
    0: "FIFO",
    1: "Earliest Due Date",
    2: "Longest Waiting Time",
    3: "Highest Lateness Risk",
}

LOGGER = logging.getLogger(__name__)


def load_hourly_arrival_rates() -> np.ndarray:
    """Load hourly order rates relative to this script."""

    if not HOURLY_RATES_PATH.exists():
        raise FileNotFoundError(
            f"Missing hourly arrival rates file: {HOURLY_RATES_PATH}"
        )
    return np.loadtxt(HOURLY_RATES_PATH) / HOUR


def bin_queue_length(x: int) -> int:
    if x == 0:
        return 0
    if x <= 3:
        return 1
    if x <= 7:
        return 2
    return 3


def bin_wip(x: int) -> int:
    if x == 0:
        return 0
    if x <= 10:
        return 1
    if x <= 25:
        return 2
    return 3


def bin_fraction(x: float) -> int:
    if x <= 0:
        return 0
    if x <= 0.25:
        return 1
    return 2


def lateness_risk(slack: float, risk_t1: float, risk_t2: float) -> float:
    """Soft lateness-risk score: 1 at/under t1, 0 at/over t2."""

    if risk_t2 <= risk_t1:
        raise ValueError("risk_t2 must be greater than risk_t1.")
    if slack <= risk_t1:
        return 1.0
    if slack >= risk_t2:
        return 0.0
    return float((risk_t2 - slack) / (risk_t2 - risk_t1))


def select_order_by_rule(queue: sim.Queue, action: int, env: sim.Environment):
    """Map an action to a dispatching rule and return the selected order."""

    orders = list(queue)
    if not orders:
        raise ValueError("Cannot select from an empty queue.")

    if action == 0:
        return min(orders, key=lambda order: order.queue_enter_time)
    if action == 1:
        return min(
            orders,
            key=lambda order: (order.starting_time + order.due_date, order.queue_enter_time),
        )
    if action == 2:
        return max(
            orders,
            key=lambda order: (env.now() - order.queue_enter_time, -order.queue_enter_time),
        )
    if action == 3:
        risk_t1 = getattr(env, "risk_t1", DEFAULT_RISK_T1)
        risk_t2 = getattr(env, "risk_t2", DEFAULT_RISK_T1 + DEFAULT_RISK_WINDOW)
        scored_orders = []
        for order in orders:
            slack = (order.starting_time + order.due_date) - env.now()
            risk = lateness_risk(slack, risk_t1, risk_t2)
            waiting_time = env.now() - order.queue_enter_time
            scored_orders.append((order, risk, waiting_time))

        if any(risk > 0 for _, risk, _ in scored_orders):
            return max(
                scored_orders,
                key=lambda item: (
                    round(item[1], 6),
                    item[2],
                    -item[0].queue_enter_time,
                ),
            )[0]

        # When no order is close to its due date, age the queue to avoid starvation
        # instead of collapsing to EDD/FIFO for all-safe orders.
        return max(
            scored_orders,
            key=lambda item: (item[2], -item[0].queue_enter_time),
        )[0]
    raise ValueError(f"Unknown action {action}. Valid actions: {sorted(ACTION_NAMES)}")


class StationQueue:
    """Explicit queue with a small wake-up mechanism for idle station servers."""

    def __init__(self, env: sim.Environment, name: str):
        self.env = env
        self.queue = sim.Queue(name)
        self.idle_servers: list[sim.Component] = []

    def __len__(self) -> int:
        return len(self.queue)

    def enqueue(self, order: "OrderRL", stage: str) -> None:
        order.current_stage = stage
        order.queue_enter_time = self.env.now()
        order.route_history.append((stage, self.env.now()))
        order.enter(self.queue)
        for server in list(self.idle_servers):
            if server.ispassive():
                server.activate()
        self.idle_servers.clear()

    def select_and_remove(self, action: int, env: sim.Environment):
        order = select_order_by_rule(self.queue, action, env)
        order.leave(self.queue)
        return order


class RLDispatcher:
    """Builds states, chooses actions and performs delayed Q-learning updates.

    MDP:
    - Decision point: Sorting server is available and Sorting queue has >= 2 orders.
    - State: (sorting_queue_bin, evaluation_queue_bin, wip_bin, due_risk_bin, rework_bin)
    - Action: choose one dispatching rule from ACTION_NAMES.
    - Reward: accumulated event rewards plus WIP and total queue penalties.
    - Episode: one complete simulation run.
    """

    def __init__(self, env: sim.Environment, agent: BaseAgent, training: bool = False):
        self.env = env
        self.agent = agent
        self.training = training
        self.previous_state: tuple[int, ...] | None = None
        self.previous_action: int | None = None

    def build_state(self) -> tuple[int, int, int, int, int]:
        wip_orders = list(self.env.orders)
        wip = len(wip_orders)
        if wip == 0:
            due_risk_bin = 0
            rework_bin = 0
        else:
            near_due = sum(
                1
                for order in wip_orders
                if (order.starting_time + order.due_date - self.env.now())
                <= DUE_RISK_THRESHOLD
            )
            rework = sum(1 for order in wip_orders if order.failed_evaluations > 0)
            due_risk_bin = bin_fraction(near_due / wip)
            rework_bin = bin_fraction(rework / wip)
        return (
            bin_queue_length(len(self.env.queue_sorting)),
            bin_queue_length(len(self.env.queue_evaluation)),
            bin_wip(wip),
            due_risk_bin,
            rework_bin,
        )

    def total_queue_length(self) -> int:
        return sum(len(queue) for queue in self.env.station_queues)

    def choose_action(self) -> int:
        self.env.rl_accumulated_reward += -0.001 * len(self.env.orders)
        self.env.rl_accumulated_reward += -0.002 * self.total_queue_length()

        current_state = self.build_state()
        if self.previous_state is not None and self.previous_action is not None:
            self.agent.observe_transition(
                self.previous_state,
                self.previous_action,
                self.env.rl_accumulated_reward,
                current_state,
                done=False,
            )

        action = self.agent.select_action(current_state, training=self.training)
        self.env.action_counts[action] += 1
        self.env.n_decision_points += 1
        self.previous_state = current_state
        self.previous_action = action
        self.env.rl_accumulated_reward = 0.0
        return action

    def finish_episode(self) -> None:
        if self.previous_state is not None and self.previous_action is not None:
            final_state = self.build_state()
            self.agent.observe_transition(
                self.previous_state,
                self.previous_action,
                self.env.rl_accumulated_reward,
                final_state,
                done=True,
            )
        self.agent.end_episode()


class OrderRL(sim.Component):
    """Order entity used by the active RL station architecture."""

    def setup(self):
        self.starting_time = self.env.now()
        self.due_date = self.env.due_date_assignment_distribution.sample()
        self.failed_evaluations = 0
        self.in_range = True
        self.queue_enter_time = self.env.now()
        self.current_stage = "created"
        self.route_history: list[tuple[str, float]] = []
        self.analysis1 = False
        self.analysis2 = False

    def process(self):
        self.enter(self.env.orders)
        self.env.queue_preparation.enqueue(self, "preparation")
        self.passivate()


class OrderSource(sim.Component):
    """Creates orders from either hourly or constant arrival rates."""

    def setup(self, use_hourly_arrival_rates: bool, rate_multiplier: float):
        self.use_hourly_arrival_rates = use_hourly_arrival_rates
        self.rate_multiplier = rate_multiplier
        self.hourly_rates = load_hourly_arrival_rates() * rate_multiplier

    def next_interarrival_time(self) -> float:
        if not self.use_hourly_arrival_rates:
            return sim.Exponential(self.rate_multiplier / CONST_ARRIVAL_RATE).sample()

        idx = math.floor(self.env.now() / HOUR) % len(self.hourly_rates)
        skipped = 0
        while self.hourly_rates[idx] == 0:
            idx = (idx + 1) % len(self.hourly_rates)
            skipped += HOUR
        return skipped + sim.Uniform(1 / self.hourly_rates[idx]).sample()

    def process(self):
        while True:
            OrderRL(env=self.env)
            self.hold(self.next_interarrival_time())


class RLStationServer(sim.Component):
    """One active server belonging to an RLStation."""

    def setup(self, station: "RLStation", server_nr: int):
        self.station = station
        self.server_nr = server_nr

    def process(self):
        while True:
            if len(self.station.queue_wrapper) == 0:
                self.station.queue_wrapper.idle_servers.append(self)
                self.passivate()
                continue

            action = 0
            if self.station.controlled_by_rl and len(self.station.queue_wrapper) >= 2:
                action = self.station.dispatcher.choose_action()

            order = self.station.queue_wrapper.select_and_remove(action, self.env)
            if self.station.n_workers > 0:
                self.request((self.env.resource_worker, self.station.n_workers))

            duration = self.station.process_time_dist.sample()
            self.hold(duration, mode=self.station.name)

            if self.station.n_workers > 0:
                self.release(self.env.resource_worker)

            self.station.next_stage_callback(order)


class RLStation:
    """Active FIFO or RL-controlled station with one or more identical servers."""

    def __init__(
        self,
        env: sim.Environment,
        name: str,
        queue_wrapper: StationQueue,
        process_time_dist,
        next_stage_callback: Callable[["OrderRL"], None],
        capacity: int = 1,
        controlled_by_rl: bool = False,
        dispatcher: RLDispatcher | None = None,
        n_workers: int = 0,
    ):
        self.env = env
        self.name = name
        self.queue_wrapper = queue_wrapper
        self.process_time_dist = process_time_dist
        self.next_stage_callback = next_stage_callback
        self.capacity = capacity
        self.controlled_by_rl = controlled_by_rl
        self.dispatcher = dispatcher
        self.n_workers = n_workers
        self.servers = [
            RLStationServer(env=env, name=f"{name}_server_{i + 1}", station=self, server_nr=i + 1)
            for i in range(capacity)
        ]


def make_agent(
    agent: BaseAgent | None,
    agent_type: str,
    fixed_action: int,
    random_seed: int | None,
    q_learning_config: dict | None,
) -> BaseAgent:
    if agent is not None:
        return agent
    if agent_type == "baseline":
        return BaselineAgent(fixed_action=fixed_action)
    if agent_type == "random":
        return RandomAgent(random_seed=random_seed)
    if agent_type == "q_learning":
        return QLearningAgent(random_seed=random_seed, **(q_learning_config or {}))
    raise ValueError("agent_type must be 'baseline', 'random' or 'q_learning'.")


def _safe_stat(values: list[float], fn: Callable[[list[float]], float]) -> float:
    return float(fn(values)) if values else float("nan")


def simulate_rl(
    agent: BaseAgent | None = None,
    agent_type: str = "baseline",
    fixed_action: int = 0,
    training: bool = False,
    run_duration: float = 1 * DAY,
    random_seed: int | str | None = None,
    animate: bool = False,
    verbose: bool = False,
    use_hourly_arrival_rates: bool = False,
    rate_multiplier: float = RATE_MULTIPLIER,
    analysis1_fraction: float = 0.8,
    analysis2_post1_fraction: float = 0.5,
    eval_failed_fraction: float = 0.1,
    worker_capacity: int = 1,
    preparation_capacity: int = 1,
    sorting_capacity: int = 1,
    analysis1_capacity: int = 1,
    analysis2_capacity: int = 1,
    evaluation_capacity: int = 1,
    dispatching_capacity: int = 1,
    preparation_pt_low: float = 1.5,
    preparation_pt_high: float = 2.5,
    preparation_pt_mode: float = 2,
    sorting_pt_low: float = 3,
    sorting_pt_high: float = 5,
    sorting_pt_mode: float = 4,
    analysis1_pt_low: float = 3,
    analysis1_pt_high: float = 6,
    analysis1_pt_mode: float = 4,
    analysis2_pt_low: float = 2,
    analysis2_pt_high: float = 2,
    analysis2_pt_mode: float = 2,
    evaluation_pt_low: float = 0.2,
    evaluation_pt_high: float = 0.75,
    evaluation_pt_mode: float = 0.4,
    dispatching_pt_low: float = 0.8,
    dispatching_pt_high: float = 1.2,
    dispatching_pt_mode: float = 1,
    risk_t1: float = DEFAULT_RISK_T1,
    risk_window: float = DEFAULT_RISK_WINDOW,
    q_learning_config: dict | None = None,
    **kwargs,
) -> dict:
    """Run one RL episode and return KPI results.

    If no orders complete, time-in-system KPIs and late_order_fraction are np.nan.
    """

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    if risk_window <= 0:
        raise ValueError("risk_window must be positive so risk_t2 is greater than risk_t1.")
    seed = "*" if random_seed is None else random_seed
    env = sim.Environment(random_seed=seed)
    env.animation_parameters(animate=animate, speed=ANIMATION_SPEED, title="Lab RL Simulation")
    env.risk_t1 = float(risk_t1)
    env.risk_window = float(risk_window)
    env.risk_t2 = env.risk_t1 + env.risk_window

    rl_agent = make_agent(agent, agent_type, fixed_action, None if seed == "*" else int(seed), q_learning_config)
    rl_agent.start_episode()

    assert 0 <= analysis1_fraction <= 1
    assert 0 <= analysis2_post1_fraction <= 1
    assert 0 <= eval_failed_fraction <= 1

    env.due_date_assignment_distribution = sim.Uniform(
        DUE_DATE_LOWER_BOUND * HOUR,
        DUE_DATE_UPPER_BOUND * HOUR,
    )
    env.analysis1_distribution = sim.Pdf(
        [True, False], probabilities=[analysis1_fraction, 1 - analysis1_fraction]
    )
    env.analysis2_post1_distribution = sim.Pdf(
        [True, False], probabilities=[analysis2_post1_fraction, 1 - analysis2_post1_fraction]
    )
    env.eval_failed_distribution = sim.Pdf(
        [True, False], probabilities=[eval_failed_fraction, 1 - eval_failed_fraction]
    )

    env.resource_worker = sim.Resource("worker", capacity=worker_capacity)
    env.orders = sim.Queue("orders")
    env.time_in_system_values = []
    env.n_orders_in_date = 0
    env.n_orders_late = 0
    env.n_eval_failures = 0
    env.total_reward = 0.0
    env.rl_accumulated_reward = 0.0
    env.n_decision_points = 0
    env.action_counts = {action: 0 for action in ACTION_NAMES}

    env.queue_preparation = StationQueue(env, "preparation_queue")
    env.queue_sorting = StationQueue(env, "sorting_queue")
    env.queue_analysis1 = StationQueue(env, "analysis1_queue")
    env.queue_analysis2 = StationQueue(env, "analysis2_queue")
    env.queue_evaluation = StationQueue(env, "evaluation_queue")
    env.queue_dispatching = StationQueue(env, "dispatching_queue")
    env.station_queues = [
        env.queue_preparation,
        env.queue_sorting,
        env.queue_analysis1,
        env.queue_analysis2,
        env.queue_evaluation,
        env.queue_dispatching,
    ]

    def add_reward(value: float) -> None:
        env.total_reward += value
        env.rl_accumulated_reward += value

    def after_preparation(order: OrderRL) -> None:
        env.queue_sorting.enqueue(order, "sorting")

    def after_sorting(order: OrderRL) -> None:
        order.analysis1 = env.analysis1_distribution.sample()
        order.analysis2 = (not order.analysis1) or env.analysis2_post1_distribution.sample()
        if order.analysis1:
            env.queue_analysis1.enqueue(order, "analysis1")
        elif order.analysis2:
            env.queue_analysis2.enqueue(order, "analysis2")
        else:
            env.queue_evaluation.enqueue(order, "evaluation")

    def after_analysis1(order: OrderRL) -> None:
        if order.analysis2:
            env.queue_analysis2.enqueue(order, "analysis2")
        else:
            env.queue_evaluation.enqueue(order, "evaluation")

    def after_analysis2(order: OrderRL) -> None:
        env.queue_evaluation.enqueue(order, "evaluation")

    def after_evaluation(order: OrderRL) -> None:
        if env.eval_failed_distribution.sample():
            order.failed_evaluations += 1
            env.n_eval_failures += 1
            add_reward(-3.0)
            env.queue_sorting.enqueue(order, "sorting_rework")
        else:
            env.queue_dispatching.enqueue(order, "dispatching")

    def after_dispatching(order: OrderRL) -> None:
        time_in_system = env.now() - order.starting_time
        env.time_in_system_values.append(time_in_system)
        reward = 10.0
        if time_in_system > order.due_date:
            order.in_range = False
            env.n_orders_late += 1
            reward -= 20.0
        else:
            env.n_orders_in_date += 1
        add_reward(reward)
        order.leave(env.orders)
        order.cancel()

    env.server_preparation_pt = sim.Triangular(preparation_pt_low * MINUTE, preparation_pt_high * MINUTE, preparation_pt_mode * MINUTE)
    env.server_sorting_pt = sim.Triangular(sorting_pt_low * MINUTE, sorting_pt_high * MINUTE, sorting_pt_mode * MINUTE)
    env.server_analysis1_pt = sim.Triangular(analysis1_pt_low * MINUTE, analysis1_pt_high * MINUTE, analysis1_pt_mode * MINUTE)
    env.server_analysis2_pt = sim.Triangular(analysis2_pt_low * MINUTE, analysis2_pt_high * MINUTE, analysis2_pt_mode * MINUTE)
    env.server_evaluation_pt = sim.Triangular(evaluation_pt_low * MINUTE, evaluation_pt_high * MINUTE, evaluation_pt_mode * MINUTE)
    env.server_dispatching_pt = sim.Triangular(dispatching_pt_low * MINUTE, dispatching_pt_high * MINUTE, dispatching_pt_mode * MINUTE)

    dispatcher = RLDispatcher(env, rl_agent, training=training)
    RLStation(env, "preparation", env.queue_preparation, env.server_preparation_pt, after_preparation, preparation_capacity)
    RLStation(env, "sorting", env.queue_sorting, env.server_sorting_pt, after_sorting, sorting_capacity, True, dispatcher)
    RLStation(env, "analysis1", env.queue_analysis1, env.server_analysis1_pt, after_analysis1, analysis1_capacity, n_workers=1)
    RLStation(env, "analysis2", env.queue_analysis2, env.server_analysis2_pt, after_analysis2, analysis2_capacity, n_workers=1)
    RLStation(env, "evaluation", env.queue_evaluation, env.server_evaluation_pt, after_evaluation, evaluation_capacity)
    RLStation(env, "dispatching", env.queue_dispatching, env.server_dispatching_pt, after_dispatching, dispatching_capacity)

    OrderSource(env=env, use_hourly_arrival_rates=use_hourly_arrival_rates, rate_multiplier=rate_multiplier)

    msg = "simulation ended"
    try:
        env.run(run_duration)
    except Exception as exc:  # Keep caller-side experiments robust.
        msg = f"simulation error: {exc}"
        LOGGER.exception("RL simulation failed")
    finally:
        dispatcher.finish_episode()

    values = env.time_in_system_values
    completed = len(values)
    late_fraction = env.n_orders_late / completed if completed else float("nan")
    epsilon = getattr(rl_agent, "epsilon", None)

    return {
        "random_seed": random_seed,
        "run_duration": run_duration,
        "due_date_lower_bound": DUE_DATE_LOWER_BOUND,
        "due_date_upper_bound": DUE_DATE_UPPER_BOUND,
        "rate_multiplier": rate_multiplier,
        "risk_t1": env.risk_t1,
        "risk_window": env.risk_window,
        "risk_t2": env.risk_t2,
        "agent_type": agent_type,
        "training": training,
        "fixed_action": fixed_action if isinstance(rl_agent, BaselineAgent) else None,
        "msg": msg,
        "n_orders_created": env.orders.number_of_arrivals,
        "n_orders_completed": completed,
        "n_orders_in_date": env.n_orders_in_date,
        "n_orders_late": env.n_orders_late,
        "late_order_fraction": late_fraction,
        "time_in_system_mean": _safe_stat(values, np.mean),
        "time_in_system_std": _safe_stat(values, np.std),
        "time_in_system_min": _safe_stat(values, np.min),
        "time_in_system_max": _safe_stat(values, np.max),
        "wip_mean": env.orders.length.mean(),
        "wip_max": env.orders.length.maximum(),
        "total_reward": env.total_reward,
        "n_eval_failures": env.n_eval_failures,
        "n_decision_points": env.n_decision_points,
        "action_0_count": env.action_counts[0],
        "action_1_count": env.action_counts[1],
        "action_2_count": env.action_counts[2],
        "action_3_count": env.action_counts[3],
        "epsilon": epsilon,
        "preparation_capacity": preparation_capacity,
        "sorting_capacity": sorting_capacity,
        "analysis1_capacity": analysis1_capacity,
        "analysis2_capacity": analysis2_capacity,
        "evaluation_capacity": evaluation_capacity,
        "dispatching_capacity": dispatching_capacity,
        "worker_capacity": worker_capacity,
    }


if __name__ == "__main__":
    result = simulate_rl(random_seed=12345, run_duration=1 * DAY, verbose=True, 
                         preparation_capacity=1, 
                         sorting_capacity=1, 
                         analysis1_capacity=1, 
                         analysis2_capacity=1, 
                         evaluation_capacity=1, 
                         dispatching_capacity=1, 
                         worker_capacity=1)
    print(result)
