from driver_dojo.core.config import Config
from driver_dojo.core.env import DriverDojoEnv
from driver_dojo.core.types import *


def get_env_config(args):
    num_maps, num_traffic, num_tasks = [int(x) for x in args.task.split('_')]

    c = Config()
    c.actions.space = 'Discretized'
    c.vehicle.dynamics_model = 'KS'
    c.simulation.dt = 0.2
    c.simulation.max_time_steps = 500
    c.scenario.traffic_init = True
    c.scenario.traffic_init_spread = 30.0
    c.scenario.traffic_spawn = True
    c.scenario.traffic_spawn_period = 1.0
    c.scenario.behavior_dist = False
    c.scenario.ego_init = True
    c.scenario.seed_offset = args.env_seed_offset
    c.scenario.seeding_mode = args.env_seeding_mode
    c.vehicle.v_max = 13.34
    c.scenario.name = 'Intersection'
    c.scenario.kwargs['crossing_style'] = 'Minor'
    c.scenario.tasks = ['L']
    c.scenario.num_maps = num_maps
    c.scenario.num_traffic = num_traffic
    c.scenario.num_tasks = num_tasks
    c.scenario.generation_threading = True

    if args.no_traffic:
        c.scenario.traffic_spawn = False
        c.scenario.traffic_init = False

    return c
