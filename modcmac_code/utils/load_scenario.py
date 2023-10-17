from ..environments.scenario import Scenario
import json
import numpy as np


def load_scenario_json(scenario_json_path: str) -> Scenario:
    """
    Loads a scenario from a json file.

    Parameters
    ----------
    scenario_json_path: str
        Path to the json file.

    Returns
    -------
    scenario: Scenario
        The scenario.
    """
    with open(scenario_json_path, 'r') as f:
        scenario_dict = json.load(f)
    if "transitions" not in scenario_dict:
        raise ValueError("Scenario file does not contain transitions")
    transitions = np.array(scenario_dict["transitions"], dtype=int)
    ncomp, timesteps = transitions.shape

    if "det_rate" not in scenario_dict:
        det_rate = np.zeros((ncomp, 1), dtype=int)
    else:
        det_rate = np.array(scenario_dict["det_rate"], dtype=int)
    if "initial_state" not in scenario_dict:
        initial_state = np.zeros(ncomp, dtype=int)
    else:
        initial_state = np.array(scenario_dict["initial_state"], dtype=int)
    if len(initial_state.shape) == 0:
        raise ValueError("Initial state is not a vector. It is a scalar.")
    if initial_state.shape[0] != ncomp:
        raise ValueError("Initial state does not have the correct shape. It is {} but should be {}".format(
            initial_state.shape[0], ncomp))
    if "initial_belief" not in scenario_dict:
        initial_belief = None
    else:
        initial_belief = np.array(scenario_dict["initial_belief"], dtype=float)
        shape_belief = initial_belief.shape[0]
        if shape_belief != ncomp:
            raise ValueError("Initial belief does not have the correct shape. It is {} but should be {}".format(
                shape_belief, ncomp))
        sum_belief_axis = np.sum(initial_belief, axis=1)
        if not np.allclose(sum_belief_axis, 1):
            raise ValueError("Initial belief does not sum to 1")
    if "name" not in scenario_dict:
        name = None
    else:
        name = scenario_dict["name"]
    if "global_action" not in scenario_dict:
        raise ValueError("Scenario file does not contain global_action. Add this to the scenario file.")
    global_action = scenario_dict["global_action"]
    scenario = Scenario(transitions=transitions, initial_state=initial_state, ncomp=ncomp, timesteps=timesteps,
                        initial_belief=initial_belief, name=name, det_rate=det_rate, global_action=global_action)
    return scenario
