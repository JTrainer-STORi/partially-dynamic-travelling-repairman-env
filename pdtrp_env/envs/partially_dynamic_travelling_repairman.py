import numpy as np

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
from matplotlib import collections as mc 

class PartiallyDynamicTravellingRepairmanEnv(gym.Env):
    """
    ## Description

    This environment simulates the partially dynamic travelling repairman problem described in the paper:
    ["Partially dynamic vehicle routing—models and algorithms"](A Larsen, O Madsen, M Solomon)
    A Repairman must visit a set of customers, each with a specific service time and location.
    The repairman starts at a depot and must service all customers while minimizing the total travel time.
    The environment allows for a fixed number of customers, and the repairman can only service a customer once.
    The environment is partially dynamic, meaning that new customers can be added during the episode.

    ## Action Space

    The action space is discrete, where each action corresponds to a customer index (1 to `max_customers`) or the depot (0).
    The repairman can only return to the depot after servicing all customers.
    The action space is defined as follows:
    - 0: Depot (starting point)
    - 1 to `max_customers`: Customer indices (1-based, excluding depot)
    The action space is defined as a discrete space with `max_customers + 1` actions, where the first action (0) is the depot.
    The repairman can only choose an action that corresponds to a customer that has arrived and has not been visited yet.
    The action mask is used to filter out invalid actions based on the current state of the environment.
    The action mask is a boolean array of length `max_customers + 1`, where each element corresponds to an action.
    If an action is valid, the corresponding element in the mask is `True`, otherwise it is `False`.
    The action mask is updated at each step based on the current time and the arrival times of the customers.

    ## Observation Space

    The customer locations exist in a 10_000 x 10_000 m^2 grid.

    The observation space is a dictionary containing:
    - `coords`: A 2D array of shape `(max_customers + 1, 2)` representing the coordinates of the depot and customers.
      The first row corresponds to the depot, and the subsequent rows correspond to the customers.
    - `repairman_location`: A 1D array of shape `(2,)` representing the current location of the repairman in the grid.

    ## Rewards 

    The reward is the negative travel time taken to service a customer.
    The reward is calculated as the negative of the travel time taken to service a customer.
    The travel time is calculated based on the distance between the repairman's current location and the customer's location,
    and the speed of the repairman.

    ## Starting State 

    The environment starts with a random number of customers (between `min_customers` and `max_customers`) and their locations.
    A number of advanced request customers (who are available immediately) and immediate request customers (who arrive at random times) are generated according to the specified maximum and minimum degrees of dynamism (`min_dod` and `max_dod`).
    The repairman starts at the depot, which is located at the center of the grid (5000, 5000).
    The service times for each customer are sampled from a log-normal distribution with a specified mean and variance.
    The arrival times of immediate request customers are sampled uniformly from the range [60, `time_horizon` * 3600] seconds.
    The arrival times of advanced request customers are set to 0 seconds, meaning they are available immediately.

    ## Episode End

    The episode ends when all customers have been serviced and the repairman returns to the depot.

    ## Arguments 

    - `min_customers`: The minimum number of customers that must be present in the environment at any time.
    - `max_customers`: The maximum number of customers that can be present in the environment at any time.
    - `repairman_speed`: The speed of the repairman in km/h.
    - `time_horizon`: The time horizon corresponding to the latest time a new customer can arrive in hours.
    - `mean_service_time`: The mean service time for each customer in minutes (min 0).
    - `var_service_time`: The variance of the service time for each customer in minutes (truncated at 0).
    - `min_dod`: The minimum degree of dynamism, which is the proportion of requests that arrive during the episode.
    - `max_dod`: The maximum degree of dynamism


    """

    metadata = {"render_modes": ["human"], "render_fps": 2}


    def __init__(self, min_customers: int = 20, max_customers: int = 100, repairman_speed: float = 40.0, time_horizon: float = 8, mean_service_time: float = 3.0, var_service_time: float = 5.0, min_dod: float = 0.1, max_dod: float = 0.9):
    
        self.size = 10_000

        self.min_customers = min_customers
        self.max_customers = max_customers
        self.repairman_speed = repairman_speed
        self.time_horizon = time_horizon
        self.mean_service_time = mean_service_time
        self.var_service_time = var_service_time
        self.min_dod = min_dod
        self.max_dod = max_dod
        self.fig, self.ax = None, None

        # Observation space is a list of customer locations and their service times. The graph representation will be constructed using a wrapper. 

        self.observation_space = spaces.Dict({
            "coords": spaces.Box(low=0, high=self.size, shape=(self.max_customers + 1, 2), dtype=np.float32),
            "repairman_location": spaces.Box(low=0, high=self.size, shape=(2,), dtype=np.float32),
        })
    
        self.action_space = spaces.Discrete(self.max_customers + 1)  # +1 for the depot

    def _sample_lognorm(self, mean, std_dev, num_samples):
        shape = np.sqrt(np.log((std_dev**2 / mean**2) + 1))
        scale = mean / np.exp(shape**2 / 2)
        return self.np_random.lognormal(mean=np.log(scale), sigma=shape, size=num_samples)

    def _generate_customers(self):
        num_customers = self.np_random.integers(self.min_customers, self.max_customers + 1) # +1 as high is exclusive
        locations = self.np_random.uniform(0, self.size, size=(num_customers, 2)).astype(np.float32)
        service_times = self._sample_lognorm(self.mean_service_time, np.sqrt(self.var_service_time), num_customers).astype(np.float32)
        return num_customers, locations, service_times
    
    def _generate_arrival_times(self):
        # Leave a minute buffer before the first customer arrives
        arrival_times = self.np_random.uniform(60, self.time_horizon * 3600, size=self.num_customers + 1).astype(np.float32)
        arrival_times[:self.n_advanced + 1] = 0.0  # Advanced request customers and depot arrive immediately
        # Sort arrival times to ensure they are in increasing order
        return np.sort(arrival_times)
    
    def action_masks(self):
        visited = set(self.tour)
        all_customers_visited = len(visited) == self.num_customers + 1  # +1 for depot

        mask = np.zeros(self.num_customers + 1, dtype=bool)
        for i in range(1, self.num_customers + 1):  # exclude depot
            if i not in visited and self.arrival_times[i] <= self.current_time:
                mask[i] = True

        # Allow depot only at end
        if all_customers_visited:
            mask[0] = True

        return mask
    
    def _get_obs(self):
        visited = np.zeros(self.num_customers + 1, dtype=np.int8)
        visited[self.tour] = 1

        return {
            "coords": self.coords,
            "repairman_location": self.repairman_location,
        }
    
    def _get_info(self):
        return {
            "current_time": self.current_time,
            "current_tour": self.tour,
            "visit_times": self.visit_times,
            "arrival_times": self.arrival_times,
            "action_mask": self.action_masks(),
        }
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.current_time = 0.0
        self.current_step = 0
        # sample the number of customers and their locations and service times
        self.num_customers, self.customer_locations, self.service_times = self._generate_customers()
        # depot location is fixed at the center of the grid
        self.depot_location = np.array([self.size / 2, self.size / 2], dtype=np.float32)
        self.coords = np.vstack((self.depot_location, self.customer_locations))  # Include depot in coords
        self.service_times = np.concatenate(([0.0], self.service_times))  # Service time for depot is 0
        # Generate the number of immediate request customers
        self.n_immediate = self.np_random.integers(int(self.min_dod * self.num_customers), int(self.max_dod * self.num_customers) + 1)
        # number of advanced request customers is therefore the total - immediate
        self.n_advanced = self.num_customers - self.n_immediate
        # Generate arrival times for customers and depot
        self.arrival_times = self._generate_arrival_times()
        self.repairman_location = self.depot_location.copy()
        self.done = False
        self.tour = [0]  # Start at the depot
        self.visit_times = [0.0]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def _travel_time(self, from_idx, to_idx):
        if from_idx == to_idx:
            return 0.0
        from_location = self.coords[from_idx]
        to_location = self.coords[to_idx]
        distance = np.linalg.norm(from_location - to_location)
        # Convert speed from km/h to m/s
        speed_m_s = (self.repairman_speed * 1000) / 3600
        travel_time = distance / speed_m_s
        return travel_time
    
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Please reset the environment.")
        
        self.current_step += 1
        
        mask = self.action_masks()

        assert mask[action], f"Invalid action {action}"

        from_idx = self.tour[-1]
        travel_time = self._travel_time(from_idx, action)
        self.visit_times.append(self.current_time + travel_time)
        self.current_time += travel_time + self.service_times[action]
        self.tour.append(action)
        self.repairman_location = self.coords[action].copy()

        # Done if tour includes all nodes AND ends at depot
        visited_set = set(self.tour)
        done = len(visited_set) == self.num_customers + 1 and self.tour[-1] == 0

        # if there are no valid actions left and we are not done, then time needs to be advanced
        if not done and not np.any(self.action_masks()):
            # Advance time to the next customer's arrival time
            remaining = [i for i in range(1, self.num_customers + 1) if i not in self.tour]
            if remaining:
                next_arrival_time = np.min(self.arrival_times[remaining])
            if next_arrival_time > self.current_time:
                self.current_time = next_arrival_time
            
        obs = self._get_obs()
        info = self._get_info()

        return obs, -travel_time, done, False, info
    
    def get_hindsight_problem(self):
        """After the episode is done, this function can be used to obtain the 'hindsight problem' i.e. as if we had future information about all customers.
        This includes additional infor containing the arrival times of customers."""
        return {
            "coords": self.coords,
            "visit_times": self.visit_times,
            "arrival_times": self.arrival_times,
        }

    def render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()
        self.ax.clear()

        # Skip index 0 (depot), get customer indices
        customer_indices = np.arange(1, self.num_customers + 1)

        # Split based on availability
        available_mask = self.arrival_times[1:] <= self.current_time
        unavailable_mask = ~available_mask

        available_customers = self.coords[1:][available_mask]
        unavailable_customers = self.coords[1:][unavailable_mask]

        # Plot depot
        self.ax.scatter(*self.coords[0], c='green', marker='s', label='Depot')

        # Plot available and unavailable customers
        if len(available_customers) > 0:
            self.ax.scatter(available_customers[:, 0], available_customers[:, 1], c='blue', label='Available')
        if len(unavailable_customers) > 0:
            self.ax.scatter(unavailable_customers[:, 0], unavailable_customers[:, 1], c='gray', label='Not Yet Arrived')

        # Label all customers with IDs (1-based)
        for i in range(1, self.num_customers + 1):
            x, y = self.coords[i]
            self.ax.text(x, y + 10, str(i), fontsize=8, ha='center')

        # Plot the repairman's tour path
        lines = []
        for i in range(1, len(self.tour)):
            a = self.coords[self.tour[i - 1]]
            b = self.coords[self.tour[i]]
            lines.append([a, b])

        if lines:
            lc = mc.LineCollection(lines, colors='red', linewidths=2)
            self.ax.add_collection(lc)

        self.ax.set_title(f"Step {self.current_step} | Time {self.current_time:.1f} sec")
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.legend()
        plt.pause(0.3)


    def close(self):
        if self.fig:
            plt.close(self.fig)