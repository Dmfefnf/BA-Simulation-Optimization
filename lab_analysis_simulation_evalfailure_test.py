import salabim as sim
import math
from base_library import (
    BasicEntity,
    ResourceStation,
    ENTITY_WIDTH,
    ENTITY_HEIGHT,
    STATION_WIDTH,
)
import pandas as pd
import numpy as np

# Time units conversion constants
MINUTE = 1
HOUR = 60 * MINUTE
DAY = 24 * HOUR


# Simulation parameters
ANIMATION_SPEED = 2

# Arrival rate configurations
USE_HOURLY_ARRIVAL_RATES = True
CONST_ARRIVAL_RATE = 500 / DAY  # Constant arrival rate of orders per day
# Hourly arrival rates, to be used if USE_HOURLY_ARRIVAL_RATES is True
HOURLY_ARRIVAL_RATES = np.loadtxt("hourly_arrival_rates.csv") / HOUR


class HourlyRateSource(sim.Component):
    """A source component that generates orders based on hourly rates."""

    def setup(self, rate_multiplier=1):
        """Setup method for initializing the hourly rates."""
        self.rates = HOURLY_ARRIVAL_RATES * rate_multiplier
        self.n_rates = len(self.rates)

    def next_interarrival_time(self):
        """Calculate the next inter-arrival time based on the current hour's rate."""
        idx = math.floor(self.env.now() / HOUR) % self.n_rates
        skip_hours = 0
        while self.rates[idx] == 0:
            idx = (idx + 1) % self.n_rates
            skip_hours += HOUR
        return skip_hours + sim.Uniform(1 / self.rates[idx]).sample()

    def process(self):
        """Process method to continuously generate orders based on the hourly rates."""
        while True:
            Order()
            self.hold(self.next_interarrival_time())


class Order(BasicEntity):
    def setup(self):
        self.duedate

    """Represents an order moving through various processing stages in the simulation."""

    def process(self):
        """Process method defining the path and actions of an order through the system."""
        # Move to and hold at each server, becoming invisible while moving and visible while processing.
        # Each step involves requesting a server, updating the fill color to indicate processing, and releasing the server afterwards.
        # Analysis logic: Orders always get analysis1 OR analysis2 (or both). If analysis1 is chosen,
        # analysis2 may additionally be performed based on analysis2_post1_fraction. If analysis1 is not chosen,
        # analysis2 is always performed.

        # Enter the system
        self.enter(self.env.orders)

        # Initial preparation stage (performed once, outside the evaluation retry loop)
        self.process_step(
            server=self.env.server_preparation,
            duration_dist=self.env.server_preparation_pt,
            n_workers=0,
            mode="preparation",
            new_color="blue",
        )

        evaluation_success = False
        while not evaluation_success:

            # Sorting stage
            self.process_step(
                server=self.env.server_sorting,
                duration_dist=self.env.server_sorting_pt,
                n_workers=0,
                mode="sorting",
                new_color="dodgerblue",
            )

            # Analysis stages, chosen based on a binary random choice
            self.analysis1 = self.env.analysis1_distribution.sample()
            self.analysis2 = (
                not self.analysis1
            ) or self.env.analysis2_post1_distribution.sample()
            if self.analysis1:
                self.process_step(
                    server=self.env.server_analysis1,
                    duration_dist=self.env.server_analysis1_pt,
                    n_workers=1,
                    mode="analysis1",
                    new_color="salmon",
                )
            if self.analysis2:
                self.process_step(
                    server=self.env.server_analysis2,
                    duration_dist=self.env.server_analysis2_pt,
                    n_workers=1,
                    mode="analysis2",
                    new_color="dimgray",
                )

            # Evaluation stage
            self.process_step(
                server=self.env.server_evaluation,
                duration_dist=self.env.server_evaluation_pt,
                n_workers=0,
                mode="evaluation",
                new_color="green",
            )

            # Check if the order failed the evaluation stage
            evaluation_success = not self.env.eval_failed_distribution.sample()

        # Dispatching stage
        self.process_step(
            server=self.env.server_dispatching,
            duration_dist=self.env.server_dispatching_pt,
            n_workers=0,
            mode="dispatching",
            new_color="greenyellow",
        )

        # Order completion, leaving the system
        self.leave(self.env.orders)

        # Move out of the screen
        self.move_and_hold(
            x1=self.env.server_dispatching.x + 300,
            y1=self.env.server_dispatching.y,
            duration=self.env.transport_duration,
            mode="moving",
        )

    def process_step(
        self,
        server,
        input_queue,
        output_queue,
        next_input_queue,
        duration_dist,
        n_workers=0,
        mode="processing",
        new_color="green",
    ):
        self.move_and_hold(
            server.x,
            server.y,
            duration=self.env.transport_duration,
            mode="moving",
        )
        self.invisible()
        self.enter(input_queue)
        self.request(server, mode="requesting")
        self.leave(input_queue)
        if n_workers > 0:
            self.request((self.env.resource_worker, n_workers), mode="requesting")
            # draw a rectangle next to the order to represent the workers
            w = sim.AnimateRectangle(
                spec=(
                    server.x + ENTITY_WIDTH,
                    server.y,
                    server.x + STATION_WIDTH,
                    server.y + ENTITY_HEIGHT,
                ),
                fillcolor="red",
            )
        self.visible()
        duration = duration_dist()
        self.update_fillcolor(new_color, duration=duration)
        self.hold(duration, mode=mode)
        if n_workers > 0:
            self.release(self.env.resource_worker)
            w.remove()
        self.enter(output_queue)
        self.release(server)

        if len(next_input_queue) < next_input_queue.capacity:
            self.leave


def simulate(
    scenario=1,
    scenario_name="",
    replication_nr=0,
    random_seed="*",
    animate=False,
    run_duration=1 * DAY,
    use_hourly_arrival_rates=True,
    rate_multiplier=1,
    analysis1_fraction=0.8,
    analysis2_post1_fraction=0.5,
    eval_failed_fraction=0.1,
    worker_capacity=1,
    preparation_capacity=1,
    sorting_capacity=1,
    analysis1_capacity=1,
    analysis2_capacity=1,
    evaluation_capacity=1,
    dispatching_capacity=1,
    preparation_pt_low=1.5,
    preparation_pt_high=2.5,
    preparation_pt_mode=2,
    sorting_pt_low=3,
    sorting_pt_high=5,
    sorting_pt_mode=4,
    analysis1_pt_low=3,
    analysis1_pt_high=6,
    analysis1_pt_mode=4,
    analysis2_pt_low=2,
    analysis2_pt_high=2,
    analysis2_pt_mode=2,
    evaluation_pt_low=0.2,
    evaluation_pt_high=0.75,
    evaluation_pt_mode=0.4,
    dispatching_pt_low=0.8,
    dispatching_pt_high=1.2,
    dispatching_pt_mode=1,
    transport_duration=1,
    **kwargs,
):
    """
    Main simulation function that sets up and runs a simulation scenario.

    Args:
        scenario (int): Identifier for the scenario.
        scenario_name (str): Name or description of the scenario.
        replication_nr (int): Identifier for the replication of this scenario.
        random_seed (int or str): Seed for the random number generator to ensure reproducibility. Use "*" for a random seed.
        animate (bool): Flag to enable or disable animation of the simulation.
        run_duration (float): Duration to run the simulation for.
        use_hourly_arrival_rates (bool): Flag to use hourly arrival rates for orders, defined in HOURLY_ARRIVAL_RATES; if False, an exponentially distributed interarrival time with constant mean is used.
        rate_multiplier (float): Multiplier for the arrival rates to adjust the overall order arrival rate.
        analysis1_fraction (float): Fraction of orders that undergo the first type of analysis.
        analysis2_post1_fraction (float): Fraction of orders that undergo analysis 2 given that they already had analysis 1. Note: all orders that skip analysis 1 will always undergo analysis 2.
        eval_failed_fraction (float): Fraction of orders that fail the evaluation stage.
        worker_capacity (int): Capacity of the worker resource.
        preparation_capacity (int): Capacity of the preparation server.
        sorting_capacity (int): Capacity of the sorting server.
        analysis1_capacity (int): Capacity of the first analysis server.
        analysis2_capacity (int): Capacity of the second analysis server.
        evaluation_capacity (int): Capacity of the evaluation server.
        dispatching_capacity (int): Capacity of the dispatching server.
        preparation_pt_low (float): Lower bound of the preparation server processing time distribution.
        preparation_pt_high (float): Upper bound of the preparation server processing time distribution.
        preparation_pt_mode (float): Mode of the preparation server processing time distribution.
        sorting_pt_low (float): Lower bound of the sorting server processing time distribution.
        sorting_pt_high (float): Upper bound of the sorting server processing time distribution.
        sorting_pt_mode (float): Mode of the sorting server processing time distribution.
        analysis1_pt_low (float): Lower bound of the first analysis server processing time distribution.
        analysis1_pt_high (float): Upper bound of the first analysis server processing time distribution.
        analysis1_pt_mode (float): Mode of the first analysis server processing time distribution.
        analysis2_pt_low (float): Lower bound of the second analysis server processing time distribution.
        analysis2_pt_high (float): Upper bound of the second analysis server processing time distribution.
        analysis2_pt_mode (float): Mode of the second analysis server processing time distribution.
        evaluation_pt_low (float): Lower bound of the evaluation server processing time distribution.
        evaluation_pt_high (float): Upper bound of the evaluation server processing time distribution.
        evaluation_pt_mode (float): Mode of the evaluation server processing time distribution.
        dispatching_pt_low (float): Lower bound of the dispatching server processing time distribution.
        dispatching_pt_high (float): Upper bound of the dispatching server processing time distribution.
        dispatching_pt_mode (float): Mode of the dispatching server processing time distribution.
        transport_duration (float): Duration for moving between stations.

    Returns:
        dict: A dictionary containing the simulation parameters and results.
    """
    # Save the model parameters for reference
    params = locals().copy()
    # print(locals())

    # Initialize the simulation environment and set animation parameters
    env = sim.Environment(random_seed=random_seed)
    env.animation_parameters(
        animate=animate,
        speed=ANIMATION_SPEED,
        title="Lab Analysis Simulation",
    )
    # env.width(env.screen_width()*0.8, adjust_x0_x1_y0=True)
    # env.height(env.screen_height()*0.8)
    env.AnimateSlider(
        x=100,
        y=100,
        vmin=0,
        vmax=64,
        resolution=1,
        v=ANIMATION_SPEED,
        label="Speed",
        action=lambda speed: env.speed(float(speed)),
        env=env,
    )

    # Probability distribution for choosing the first type of analysis
    assert 0 <= analysis1_fraction <= 1
    env.analysis1_distribution = sim.Pdf(
        [True, False],
        probabilities=[analysis1_fraction, 1 - analysis1_fraction],
    )

    # Probability distribution for analysis 2 after analysis 1
    assert 0 <= analysis2_post1_fraction <= 1
    env.analysis2_post1_distribution = sim.Pdf(
        [True, False],
        [analysis2_post1_fraction, 1 - analysis2_post1_fraction],
    )

    # Probability distribution for evaluation failure
    assert 0 <= eval_failed_fraction <= 1
    env.eval_failed_distribution = sim.Pdf(
        [True, False],
        probabilities=[eval_failed_fraction, 1 - eval_failed_fraction],
    )

    # Setup resources with given capacities and positions
    env.server_preparation = ResourceStation(
        name="preparation",
        capacity=preparation_capacity,
        x=100,
        y=300,
        display_name="Preparation",
    )
    env.server_sorting = ResourceStation(
        name="sorting",
        capacity=sorting_capacity,
        x=250,
        y=300,
        display_name="Sorting",
    )
    env.server_analysis1 = ResourceStation(
        name="Analysis1",
        capacity=analysis1_capacity,
        x=400,
        y=100,
        display_name="Analysis1",
        fillcolor="plum",
    )
    env.server_analysis2 = ResourceStation(
        name="Analysis2",
        capacity=analysis2_capacity,
        x=400,
        y=500,
        display_name="Analysis2",
        fillcolor="slategray",
    )
    env.server_evaluation = ResourceStation(
        name="Evaluation",
        capacity=evaluation_capacity,
        x=550,
        y=300,
        display_name="Evaluation",
    )
    env.server_dispatching = ResourceStation(
        name="Dispatching",
        capacity=dispatching_capacity,
        x=700,
        y=300,
        display_name="Dispatching",
    )
    env.resource_worker = ResourceStation(
        name="Worker",
        capacity=worker_capacity,
        x=400,
        y=300,
        width=100,
        height=50,
        fillcolor="red",
        display_name="Worker",
    )
    env.sorting_input_queue = sim.Queue("sorting_input_queue", capacity=4)
    env.sorting_output_queue = sim.Queue("sorting_output_queue", capacity=4)

    # Define processing times for each station using distributions
    env.server_preparation_pt = sim.Triangular(
        low=preparation_pt_low * MINUTE,
        high=preparation_pt_high * MINUTE,
        mode=preparation_pt_mode * MINUTE,
    )
    env.server_sorting_pt = sim.Triangular(
        low=sorting_pt_low * MINUTE,
        high=sorting_pt_high * MINUTE,
        mode=sorting_pt_mode * MINUTE,
    )
    env.server_analysis1_pt = sim.Triangular(
        low=analysis1_pt_low * MINUTE,
        high=analysis1_pt_high * MINUTE,
        mode=analysis1_pt_mode * MINUTE,
    )
    env.server_analysis2_pt = sim.Triangular(
        low=analysis2_pt_low * MINUTE,
        high=analysis2_pt_high * MINUTE,
        mode=analysis2_pt_mode * MINUTE,
    )
    env.server_evaluation_pt = sim.Triangular(
        low=evaluation_pt_low * MINUTE,
        high=evaluation_pt_high * MINUTE,
        mode=evaluation_pt_mode * MINUTE,
    )
    env.server_dispatching_pt = sim.Triangular(
        low=dispatching_pt_low * MINUTE,
        high=dispatching_pt_high * MINUTE,
        mode=dispatching_pt_mode * MINUTE,
    )
    env.transport_duration = transport_duration  # Duration for moving between stations

    # Setup queues for monitoring and collecting statistics
    env.orders = sim.Queue("orders")

    # Initialize the source of orders based on the configuration
    if use_hourly_arrival_rates:
        HourlyRateSource(env=env, rate_multiplier=rate_multiplier)
    else:
        sim.ComponentGenerator(
            Order, iat=sim.Exponential(rate_multiplier / CONST_ARRIVAL_RATE)
        )

    # Run the simulation
    try:
        env.run(run_duration)
    except sim.SimulationStopped:
        msg = "simulation stopped"
    except Exception as e:
        msg = f"another exception: {e}"
    else:
        msg = "simulation ended"

    # Collect and return simulation results
    return {
        **params,
        "msg": msg,
        "t_end": env.now(),
        "preparation_waiting_time_mean": env.server_preparation.requesters().length_of_stay.mean(),
        "preparation_waiting_time_max": env.server_preparation.requesters().length_of_stay.maximum(),
        "preparation_queue_length_mean": env.server_preparation.requesters().length.mean(),
        "preparation_queue_length_max": env.server_preparation.requesters().length.maximum(),
        "preparation_occupancy_mean": env.server_preparation.occupancy.mean(),
        "sorting_waiting_time_mean": env.server_sorting.requesters().length_of_stay.mean(),
        "sorting_waiting_time_max": env.server_sorting.requesters().length_of_stay.maximum(),
        "sorting_queue_length_mean": env.server_sorting.requesters().length.mean(),
        "sorting_queue_length_max": env.server_sorting.requesters().length.maximum(),
        "analysis1_waiting_time_mean": env.server_analysis1.requesters().length_of_stay.mean(),
        "analysis1_waiting_time_max": env.server_analysis1.requesters().length_of_stay.maximum(),
        "analysis1_queue_length_mean": env.server_analysis1.requesters().length.mean(),
        "analysis1_queue_length_max": env.server_analysis1.requesters().length.maximum(),
        "analysis1_orders_processed": env.server_analysis1.claimers().number_of_departures,
        "analysis2_waiting_time_mean": env.server_analysis2.requesters().length_of_stay.mean(),
        "analysis2_queue_length_mean": env.server_analysis2.requesters().length.mean(),
        "analysis2_orders_processed": env.server_analysis2.claimers().number_of_departures,
        "evaluation_waiting_time_mean": env.server_evaluation.requesters().length_of_stay.mean(),
        "evaluation_waiting_time_max": env.server_evaluation.requesters().length_of_stay.maximum(),
        "evaluation_queue_length_mean": env.server_evaluation.requesters().length.mean(),
        "evaluation_queue_length_max": env.server_evaluation.requesters().length.maximum(),
        "dispatching_waiting_time_mean": env.server_dispatching.requesters().length_of_stay.mean(),
        "dispatching_waiting_time_max": env.server_dispatching.requesters().length_of_stay.maximum(),
        "dispatching_queue_length_mean": env.server_dispatching.requesters().length.mean(),
        "dispatching_queue_length_max": env.server_dispatching.requesters().length.maximum(),
        "queue_preparation_length_tx": [
            list(elem) for elem in env.server_preparation.requesters().length.tx()
        ],
        "queue_preparation_length_resampled": (
            env.server_preparation.requesters()
            .length.as_resampled_dataframe(delta_t=1 * HOUR)
            .values.tolist()
        ),
        "n_orders_created": env.orders.number_of_arrivals,
        "n_orders_completed": env.orders.number_of_departures,
        "n_orders_incomplete": (
            env.orders.number_of_arrivals - env.orders.number_of_departures
        ),
        "time_in_system_max": env.orders.length_of_stay.maximum(),
        "time_in_system_mean": env.orders.length_of_stay.mean(),
        "work_in_progress_max": env.orders.length.maximum(),
        "work_in_progress_mean": env.orders.length.mean(),
    }


def load_scenarios(scenario_file_path="scenarios.xlsx", sheet_name="scenarios"):
    """Load simulation scenarios from a CSV or Excel file.

    Args:
        scenario_file_path (str): Path to the scenarios file.
        sheet_name (str): Name of the sheet to read from (for Excel files).

    Returns:
        pd.DataFrame: DataFrame containing the scenarios.
    """
    if scenario_file_path.lower().endswith(".csv"):
        df = pd.read_csv(scenario_file_path)
    elif scenario_file_path.lower().endswith(".xlsx"):
        df = pd.read_excel(scenario_file_path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    return df


def load_scenarios_transposed(
    scenario_file_path="scenarios_transposed.xlsx",
    sheet_name="scenarios",
):
    """Load and transpose simulation scenarios from a file.

    Args:
        scenario_file_path (str): Path to the scenarios file.
        sheet_name (str): Name of the sheet to read from (for Excel files).

    Returns:
        pd.DataFrame: Transposed DataFrame containing the scenarios.
    """
    df = load_scenarios(scenario_file_path, sheet_name)
    arr = df.loc[:, "Key":].values.T
    return pd.DataFrame(arr[1:], columns=arr[0])


def run_all_scenarios(
    scenarios_file_path="scenarios_transposed.xlsx",
    results_output_path="results.xlsx",
    n_replications=20,
    starting_seed=424242,
    seed_step=5,
):
    """Run multiple simulation scenarios with replications and save results.

    Args:
        scenarios_file_path (str): Path to the scenarios file.
        results_output_path (str): Path where results will be saved.
        n_replications (int): Number of replications for each scenario.
        starting_seed (int or str): Starting seed for random number generation. Use "*" for random seeds.
        seed_step (int): Increment between seeds for different replications.

    Returns:
        pd.DataFrame: DataFrame containing all simulation results.
    """
    df = load_scenarios_transposed(scenarios_file_path)
    parameters = [
        {
            **scenario,
            "replication_nr": i,
            "random_seed": (
                starting_seed + seed_step * i if isinstance(starting_seed, int) else "*"
            ),
            "animate": False,
        }
        for scenario in df.to_dict(orient="records")
        for i in range(n_replications)
    ]
    results = pd.DataFrame([simulate(**params) for params in parameters])
    if results_output_path.lower().endswith(".csv"):
        results.to_csv(results_output_path, index=False)
    elif results_output_path.lower().endswith(".xlsx"):
        results.to_excel(results_output_path, index=False)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    return results


def run_single_scenario(
    scenarios_file_path="scenarios_transposed.xlsx",
    scenario_nr=1,
    animate=True,
    **kwargs,
):
    """Run a single simulation scenario.

    Args:
        scenarios_file_path (str): Path to the scenarios file.
        scenario_nr (int): Scenario number to run.
        animate (bool): Whether to enable animation.
        **kwargs: Additional parameters to override scenario defaults.

    Returns:
        pd.Series: Series containing the simulation results.
    """
    df = load_scenarios_transposed(scenarios_file_path)
    scenarios = df[df["scenario"] == scenario_nr]
    if len(scenarios) == 0:
        raise ValueError(f"No scenarios found for scenario number {scenario_nr}.")
    elif len(scenarios) > 1:
        print(f"Warning: Multiple scenarios found for scenario number {scenario_nr}.")
        print("Running the first one.")
    parameters = {**scenarios.iloc[0].to_dict(), "animate": animate, **kwargs}
    return pd.Series(simulate(**parameters))


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt

    # ###########################################################################
    # # Run a single scenario with animation
    # ###########################################################################
    print(run_single_scenario())

    # ###########################################################################
    # ## Run a single scenario without animation
    # ###########################################################################
    # print(run_single_scenario(animate=False))

    ###########################################################################
    # Run all scenarios (only possible without animation)
    ###########################################################################
    # df = run_all_scenarios(starting_seed="*", n_replications=10)
    # df["worker_capacity"] = df["worker_capacity"].astype(int).astype(str)
    # g = sns.catplot(
    #     data=df,
    #     y="time_in_system_mean",
    #     x="eval_failed_fraction",
    #     hue="analysis2_post1_fraction",
    #     col="worker_capacity",
    #     kind="box",
    # )
    # plt.show()
