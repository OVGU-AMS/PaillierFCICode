"""

"""

import numpy as np
from phe import paillier
import estimation as est

NUM_NODES = 16
NODE_GRID_DIST = 10
NODE_POSITIONS = [np.array([NODE_GRID_DIST*(x+1),NODE_GRID_DIST*(y+1)]) for x in range(int(NUM_NODES**0.5)) for y in range(int(NUM_NODES**0.5))]
NODE_RANGE = 12
NODE_CONNECTIONS = np.array([np.array([1 if np.linalg.norm(NODE_POSITIONS[c]-NODE_POSITIONS[r])<NODE_RANGE else 0 for c in range(NUM_NODES)]) for r in range(NUM_NODES)])

SIM_STEPS = 50
SIM_RUNS = 2

#print(NODE_POSITIONS)
#print(NODE_CONNECTIONS)

class SensorNode():
    def __init__(self, ident, position, ready_sensor, ready_filter, public_key):
        # Info
        self.id = ident
        self.position = position
        self.sensor = ready_sensor
        self.filter = ready_filter

        # Current estimation
        self.curr_est = None
        self.curr_cov = None
        self.enc_curr_est = None
        self.enc_curr_cov = None

        # Current fusion
        self.curr_fused_est = None
        self.curr_fused_cov = None
        self.enc_curr_fused_est = None
        self.enc_curr_fused_cov = None

        # Encryption
        self.pk = public_key
        return
    
    def make_local_est(self, gt):
        # Get local measurement
        z = self.sensor.measure(gt)
        
        # Get local estimate
        if self.filter:
            self.filter.predict()
            x, P = self.filter.update(gt)
        else:
            x = self.sensor.H@z
            P = self.sensor.R
        self.curr_est = x
        self.curr_cov = P
        
        # Encryption
        self.enc_curr_est = np.array([self.pk.encrypt(x) for x in self.curr_est])
        self.enc_curr_cov = np.array([[self.pk.encrypt(x) for x in row] for row in self.curr_cov])
        return

    
    
    def fuse_with(self, neighbours):
        # Communication steps of the Paillier CI algorithm
        invtrs = [n._get_curr_invtr() for n in neighbours]
        invtrs.append(self._get_curr_invtr())
        sum_invtrs = sum(invtrs)
        weighted_ests_and_covs = [n._get_weighted_est_from_sum_invtr(sum_invtrs) for n in neighbours]
        weighted_ests_and_covs.append(self._get_weighted_est_from_sum_invtr(sum_invtrs))

        # Communication for normal CI (for comparison)

        # Store plaintext of fusion

        # Store encryption of fusion
        self.curr_fused_est = sum([x[0] for x in weighted_ests_and_covs])
        self.curr_fused_cov = sum([x[1] for x in weighted_ests_and_covs])
        return
    
    def _get_curr_invtr(self):
        return self.pk.encrypt(1.0/np.trace(self.curr_cov))
    
    def _get_weighted_est_from_sum_invtr(self, sum_invtr):
        w_est = np.trace(self.curr_cov)*self.curr_est
        w_cov = np.trace(self.curr_cov)*self.curr_cov
        return sum_invtr*w_est, sum_invtr*w_cov


class Navigator():
    def __init__(self, secret_key):
        # What to store
        self.sim_state_ests = []
        self.sim_state_covs = []
        self.sim_errors = []
        self.sk = secret_key
        return
    
    def save_est_and_error(self, enc_est, enc_cov, gt):
        # Decryption

        # Stores estimate and its error at each step as simulation progresses
        self.sim_state_ests.append()
        self.sim_state_covs.append()
        self.sim_errors.append(np.linalg.norm(est-gt))
        return

def plot_sim_layout():
    return

def plot_avg_sim_errors(nav_list):

    return

def main():
    # State dimension
    n = 4

    # Measurement dimension
    m = 2

    # Process model (q = noise strength, t = timestep)
    q = 0.01
    t = 0.5
    F = np.array([[1, t, 0, 0], 
                  [0, 1, 0, 0], 
                  [0, 0, 1, t], 
                  [0, 0, 0, 1]])

    Q = q*np.array([[t**3/3, t**2/2,      0,      0], 
                    [t**2/2,      t,      0,      0], 
                    [     0,      0, t**3/3, t**2/2],
                    [     0,      0, t**2/2,      t]])

    # Measurement model
    H = np.array([[1, 0, 0, 0], 
                  [0, 0, 1, 0]])

    R = np.array([[5, 2], 
                  [2, 5]])

    # Filter init
    init_state = np.array([0, 1, 0, 1])
    init_cov = np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0], 
                         [0, 0, 0, 1]])
    
    # Ground truth init
    mean = np.array([0, 1, 0, 1])
    cov = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    
    # Encryption keys (generated once for speed)
    pk,sk = paillier.generate_paillier_keypair()

    # Result storage
    sim_navigators = []
    
    # Loop sims
    for _ in range(SIM_RUNS):

        # Ground truth
        gt_init_state = np.random.multivariate_normal(mean, cov)
        ground_truth = est.GroundTruth(F, Q, gt_init_state)

        # Nodes
        nodes = []
        for i in range(NUM_NODES):
            s = est.SensorPure(n, m, H, R)
            f = est.KFilter(n, m, F, Q, H, R, init_state, init_cov)
            nodes.append(SensorNode(i, NODE_POSITIONS[i], s, f, pk))
        
        # Navigator (and results)
        nav = Navigator(sk)
        sim_navigators.append(nav)
        
        # Start sim
        for k in range(SIM_STEPS):

            # Each node measures and/or estimates
            gt = ground_truth.update()
            for n in nodes:
                n.make_local_est(gt)
        
            # Each node fuses local with neighbours
            for n in nodes:
                neighbours = [x for x,i in enumerate(nodes) if NODE_CONNECTIONS[n.ident][i] == 1]
                n.fuse_with(neighbours)

            # Navigator takes closest fusion
            closest_node = None
            min_dist = NODE_GRID_DIST*NUM_NODES
            for n in nodes:
                dist = np.linalg.norm(n.position - gt)
                if dist < min_dist:
                    closest_node = n
            nav.save_est_and_error(closest_node.curr_fused_est, closest_node.curr_fused_cov, gt)
    
    # Plotting
    plot_avg_sim_errors(sim_navigators)

    return


if __name__ == '__main__':
    main()