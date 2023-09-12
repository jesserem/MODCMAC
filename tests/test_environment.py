from modcmac_code.environments.Maintenance_Gym import MaintenanceEnv as maintenance_env
import numpy as np


def create_env():
    ncomp = 13  # number of components
    ndeterioration = 50  # number of deterioration steps
    ntypes = 3  # number of component types
    nstcomp = 5  # number of states per component
    naglobal = 2  # number of actions global (inspect X purpose)
    nacomp = 3  # number of actions per component
    nobs = 5  # number of observations
    nfail = 3  # number of failure types

    """
    P: transition probability matrix, with dimensions (ndeterioration, ntypes, nstcomp, nstcomp)
    P_start: initial transition probability matrix, with dimensions (ntypes, nstcomp, nstcomp)
    P_end: final transition probability matrix, with dimensions (ntypes, nstcomp, nstcomp)

    The first dimension of P is the deterioration mode, which linear deteriorates from P_start to P_end
    """

    P_start = np.zeros((ntypes, nstcomp, nstcomp))
    P_start[0] = np.array([
        [0.983, 0.0089, 0.0055, 0.0025, 0.0001],
        [0, 0.9836, 0.0084, 0.0054, 0.0026],
        [0, 0, 0.9862, 0.0084, 0.0054],
        [0, 0, 0, 0.9917, 0.0083],
        [0, 0, 0, 0, 1]
    ])
    P_start[1] = np.array([[0.9748, 0.013, 0.0081, 0.004, 0.0001],
                           [0., 0.9754, 0.0124, 0.0081, 0.0041],
                           [0., 0., 0.9793, 0.0125, 0.0082],
                           [0., 0., 0., 0.9876, 0.0124],
                           [0., 0., 0., 0., 1.]])

    P_start[2] = np.array([[0.9848, 0.008, 0.0049, 0.0022, 0.0001],
                           [0., 0.9854, 0.0074, 0.0048, 0.0024],
                           [0., 0., 0.9876, 0.0075, 0.0049],
                           [0., 0., 0., 0.9926, 0.0074],
                           [0., 0., 0., 0., 1.]])

    P_end = np.zeros((ntypes, nstcomp, nstcomp))
    P_end[0] = np.array([
        [0.9713, 0.0148, 0.0093, 0.0045, 0.0001],
        [0., 0.9719, 0.0142, 0.0093, 0.0046],
        [0, 0, 0.9753, 0.0153, 0.0094],
        [0., 0., 0., 0.9858, 0.0142],
        [0., 0., 0., 0., 1.]
    ])

    P_end[1] = np.array([[0.9534, 0.0237, 0.0153, 0.0075, 0.0001],
                         [0., 0.954, 0.0231, 0.0152, 0.0077],
                         [0., 0., 0.9613, 0.0233, 0.0154],
                         [0., 0., 0., 0.9767, 0.0233],
                         [0., 0., 0., 0., 1.]])

    P_end[2] = np.array([[0.9748, 0.013, 0.0081, 0.004, 0.0001],
                         [0., 0.9754, 0.0124, 0.0081, 0.0041],
                         [0., 0., 0.9793, 0.0125, 0.0082],
                         [0., 0., 0., 0.9876, 0.0124],
                         [0., 0., 0., 0., 1.]])

    """
    Check if each row in P_start and P_end sums to 1
    """
    for i in range(ntypes):
        for j in range(nstcomp):
            P_start[i, j, :] = P_start[i, j, :] / np.sum(P_start[i, j, :])
            P_end[i, j, :] = P_end[i, j, :] / np.sum(P_end[i, j, :])

    P = np.zeros((ndeterioration, P_start.shape[0], P_start.shape[1], P_start.shape[2]))
    for i in range(ndeterioration):
        P[i, :, :] = P_start + (P_end - P_start) * i / (ndeterioration - 1)

    """
    Observation matrix
    O_no: observation matrix for the no-inspection action
    O_in: observation matrix for the inspection action
    O is the observation matrix for the inspect, no-inspect and replace action
    """

    O_in = np.eye(nstcomp)
    O_no = np.array([[1, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0],
                     [0, 0, 0.34, 0.33, 0.33],
                     [0, 0, 0.34, 0.33, 0.33],

                     [0, 0, 0.34, 0.33, 0.33]])

    O = np.zeros((2, nstcomp, nstcomp))
    O[0] = O_no
    O[1] = O_in

    repair_per = 0.25
    inspect_per = 0.015

    """
    Set the start state of the components
    0: No deterioration
    1: Small deterioration
    2: Large deterioration
    3: Near failure
    """
    start_state = np.zeros(ncomp, dtype=int)
    # Wooden Poles (index 0-8)
    start_state[:9] = np.array([3, 3, 2, 3, 2, 2, 3, 2, 3])
    # Wooden Kesp (index 9-11)
    start_state[9:12] = np.array([2, 3, 2])
    # Wooden Floor (index 12)
    start_state[12] = np.array([2])
    start_S = np.zeros((ncomp, nstcomp))
    start_S[np.arange(ncomp), start_state] = 1

    """
    TYPE 1: Wooden Pole, N=9, 40% of total cost
    TYPE 2: Wooden kesp, N=3, 3.75% of total cost
    TYPE 3: Wooden floor, N=1, 11.25% of total cost
    """

    total_cost = 1
    inspect_cost = 0.005

    n_type1 = 9
    total_cost_type1 = 0.4 * total_cost
    repla_cost_type1 = total_cost_type1 / n_type1
    n_type2 = 3
    total_cost_type2 = 0.0375 * total_cost
    repla_cost_type2 = total_cost_type2 / n_type2
    n_type3 = 1
    total_cost_type3 = 0.1125 * total_cost
    repla_cost_type3 = total_cost_type3 / n_type3

    C_glo = np.zeros((1, naglobal))
    C_glo[0] = np.array([0, inspect_cost * total_cost])

    C_rep = np.zeros((ntypes, nacomp))
    C_rep[0] = np.array([0, repair_per * repla_cost_type1, repla_cost_type1])
    C_rep[1] = np.array([0, repair_per * repla_cost_type2, repla_cost_type2])
    C_rep[2] = np.array([0, repair_per * repla_cost_type3, repla_cost_type3])

    """
    Components that will be used for the simulation
    Comp: 0, 1 and 2, Wooden Pole connected to Wooden Kesp (9)
    Comp: 3, 4 and 5, Wooden Pole connected to Wooden Kesp (10)
    Comp: 6, 7 and 8, Wooden Pole connected to Wooden Kesp (11)
    Comp: 9 Wooden Kesp connected to Wooden Floor (12)
    Comp: 10 Wooden Kesp connected to Wooden Floor (12)
    Comp: 11 Wooden Kesp connected to Wooden Floor (12)
    Comp: 12 Wooden Floor
    """
    comp_setup = np.array(([0] * 9) + ([1] * 3) + [2])

    """
    Failure Mode 1: Wooden Pole Failure. 3 substructures (0, 1, 2), (3, 4, 5), (6, 7, 8)
    """
    f_mode_1 = np.zeros((3, 3), dtype=int)
    f_mode_1[0] = np.array([0, 1, 2])
    f_mode_1[1] = np.array([3, 4, 5])
    f_mode_1[2] = np.array([6, 7, 8])

    """
    Failure Mode 2: Wooden Kesp Failure. 2 substructures (9, 10), (10, 11)
    """
    f_mode_2 = np.zeros((2, 2), dtype=int)
    f_mode_2[0] = np.array([9, 10])
    f_mode_2[1] = np.array([10, 11])

    """
    Failure Mode 3: Wooden Floor Failure. 1 substructures (12)
    """
    f_mode_3 = np.zeros((1, 1), dtype=int)
    f_mode_3[0] = np.array([12])

    f_modes = (f_mode_1, f_mode_2, f_mode_3)
    test_env = maintenance_env(ncomp, ndeterioration, ntypes, nstcomp, naglobal, nacomp, nobs, nfail, P,
                               O, C_glo, C_rep, comp_setup, f_modes, start_S, total_cost)
    return test_env


def test_env_init():
    test_env = create_env()
    assert test_env.ncomp == 13, "Number of components is not correct"
    assert test_env.ndeterioration == 50, "Number of deterioration steps is not correct"
    assert test_env.ntypes == 3, "Number of component types is not correct"
    assert test_env.nstcomp == 5, "Number of states per component is not correct"
    assert test_env.naglobal == 2, "Number of global actions is not correct"
    assert test_env.nacomp == 3, "Number of component actions is not correct"
    assert test_env.nobs == 5, "Number of observations is not correct"
    assert test_env.nfail == 3, "Number of failure modes is not correct"


def test_env_state():
    test_env = create_env()
    assert test_env.state.shape == (13, 5), "State shape is not correct"
    assert test_env.state.dtype == np.float64, "State dtype is not correct"
    assert test_env.state[0, 0] == 0, "State is not initialized correctly"


def test_reset_env_state():
    test_env = create_env()
    test_env.reset()
    assert test_env.state.shape == (13, 5), "State shape is not correct"
    assert np.equal(test_env.state, test_env.start_S).all(), "State is not reset correctly"
    assert np.equal(test_env.det_rate, np.zeros((13, 1), dtype=int)).all(), \
        "Deterioration rate is not reset correctly"




