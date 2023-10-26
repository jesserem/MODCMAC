import numpy as np


def get_quay_wall_config():
    config = dict()
    ncomp = 13  # number of components
    ndeterioration = 50  # number of deterioration steps
    ntypes = 3  # number of component types
    nstcomp = 5  # number of states per component
    naglobal = 2  # number of actions global (inspect X purpose)
    nacomp = 3  # number of actions per component
    nobs = 5  # number of observations
    nfail = 3  # number of failure types
    config['ncomp'] = ncomp
    config['ndeterioration'] = ndeterioration
    config['ntypes'] = ntypes
    config['nstcomp'] = nstcomp
    config['naglobal'] = naglobal
    config['nacomp'] = nacomp
    config['nobs'] = nobs
    config['nfail'] = nfail

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

    config['P'] = P
    # """
    # F: failure probability matrix, with dimensions (ntypes, nstcomp)
    #
    # F is the probability of failure for each component type given the current state, if failed the component stays failed
    # until replaced
    # """
    # F = np.zeros((ntypes, nstcomp))
    # F[0] = np.array([0.0008, 0.0046, 0.0123, 0.0259, 1])
    # F[1] = np.array([0.0012, 0.0073, 0.0154, 0.0324, 1])
    # F[2] = np.array([0.0019, 0.0067, 0.0115, 0.0177, 1])

    """
    Observation matrix
    O_no: observation matrix for the no-inspection action
    O_in: observation matrix for the inspection action
    O is the observation matrix for the inspect, no-inspect and replace action
    """

    # O_no = np.array([[1, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0],
    #                  [0, 0.25, 0.5, 0.25, 0],
    #                  [0, 0, 0.5, 0.5, 0],
    #                  [0, 0, 0, 0, 1]])
    # O_no = np.array([[1, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 1]])
    O_in = np.eye(nstcomp)
    # O_no = np.array([[1, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0],
    #                  [0, 0, 0.34, 0.33, 0.33],
    #                  [0, 0, 0.34, 0.33, 0.33],
    #
    #                  [0, 0, 0.34, 0.33, 0.33]])
    O_no = np.array([[1, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 1]])

    O = np.zeros((2, nstcomp, nstcomp))
    O[0] = O_no
    O[1] = O_in
    config['O'] = O
    repair_per = 0.25
    inspect_per = 0.05

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
    config['start_S'] = start_S

    """
    TYPE 1: Wooden Pole, N=9, 40% of total cost
    TYPE 2: Wooden kesp, N=3, 3.75% of total cost
    TYPE 3: Wooden floor, N=1, 11.25% of total cost
    """

    total_cost = 1
    config['total_cost'] = total_cost
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
    config['C_glo'] = C_glo

    C_rep = np.zeros((ntypes, nacomp))
    C_rep[0] = np.array([0, repair_per * repla_cost_type1, repla_cost_type1])
    C_rep[1] = np.array([0, repair_per * repla_cost_type2, repla_cost_type2])
    C_rep[2] = np.array([0, repair_per * repla_cost_type3, repla_cost_type3])
    config['C_rep'] = C_rep

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
    config['comp_setup'] = comp_setup

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
    config['f_modes'] = f_modes
    config['ep_length'] = 50

    return config
