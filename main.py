"""

"""


import numpy as np
import matplotlib.pyplot as plt
from phe import paillier
import estimation as est
import encryption as enc
import plotting as plot


NUM_NODES = 16
NODE_GRID_DIST = 10
NODE_POSITIONS = [np.array([NODE_GRID_DIST*(x+1),NODE_GRID_DIST*(y+1)]) for x in range(int(NUM_NODES**0.5)) for y in range(int(NUM_NODES**0.5))]
NODE_RANGE = 12
NODE_CONNECTIONS = np.array([np.array([1 if np.linalg.norm(NODE_POSITIONS[c]-NODE_POSITIONS[r])<NODE_RANGE else 0 for c in range(NUM_NODES)]) for r in range(NUM_NODES)])

SIM_STEPS = 20
SIM_RUNS = 2

SHOW_LAYOUT_PLOT = False
SHOW_ERROR_PLOT = False


class SensorNode():
    def __init__(self, ident, position, ready_sensor, ready_filter, public_key):
        # Info
        self.ident = ident
        self.position = position
        self.sensor = ready_sensor
        self.filter = ready_filter

        # Current estimation
        self.curr_est = None
        self.curr_cov = None
        self.curr_inf_vec = None
        self.curr_inf_mat = None
        self.enc_curr_inf_vec = None
        self.enc_curr_inf_mat = None

        # Current fusion
        self.curr_fused_inf_vec = None
        self.curr_fused_inf_mat = None
        self.enc_curr_fused_inf_vec = None
        self.enc_curr_fused_inf_mat = None
        self.enc_curr_fuse_scale = None

        # Encryption
        self.pk = public_key
        return
    
    def make_local_est(self, gt):
        # Get local measurement
        z = self.sensor.measure(gt)
        
        # Get local estimate
        self.filter.predict()
        x, P = self.filter.update(z)

        self.curr_est = x
        self.curr_cov = P
        self.curr_inf_mat = np.linalg.inv(self.curr_cov)
        self.curr_inf_vec = self.curr_inf_mat@self.curr_est
        
        # Encryption
        self.enc_curr_inf_mat = np.array([[enc.EncryptedEncoding(self.pk, x) for x in row] for row in self.curr_inf_mat])
        self.enc_curr_inf_vec = np.array([enc.EncryptedEncoding(self.pk, x) for x in self.curr_inf_vec])
        return z

    def fuse_with(self, neighbours):
        # Compute scale of weighted estimates
        trs = [n._get_curr_tr() for n in neighbours]
        trs.append(self._get_curr_tr())
        self.enc_curr_fuse_scale = sum(trs)

        # Computed weighted fused estimates
        weighted_inf_vecs_mats = [n._get_weighted_inf_from_sum_invtr() for n in neighbours]
        weighted_inf_vecs_mats.append(self._get_weighted_inf_from_sum_invtr())
        self.enc_curr_fused_inf_mat = sum([x[1] for x in weighted_inf_vecs_mats])
        self.enc_curr_fused_inf_vec = sum([x[0] for x in weighted_inf_vecs_mats])

        # Get plaintext of fusion from normal CI (for comparison)
        all_infs = [n._get_curr_plain_inf() for n in neighbours]
        all_infs.append(self._get_curr_plain_inf())
        all_cov_traces = [np.trace(np.linalg.inv(e[1])) for e in all_infs]
        trace_sum = sum(all_cov_traces)
        self.curr_fused_inf_mat = sum((all_cov_traces[i]/trace_sum) * e[1] for i,e in enumerate(all_infs))
        self.curr_fused_inf_vec = sum((all_cov_traces[i]/trace_sum) * e[0] for i,e in enumerate(all_infs))
        return
    
    def _get_curr_plain_inf(self):
        return self.curr_inf_vec, self.curr_inf_mat
    
    def _get_curr_tr(self):
        return enc.EncryptedEncoding(self.pk, np.trace(self.curr_cov))
    
    def _get_weighted_inf_from_sum_invtr(self):
        w_mat = np.trace(self.curr_cov)*self.curr_inf_mat
        w_vec = np.trace(self.curr_cov)*self.curr_inf_vec
        return np.array([enc.EncryptedEncoding(self.pk, e) for e in w_vec]), np.array([[enc.EncryptedEncoding(self.pk, e) for e in row] for row in w_mat])


class Navigator():
    def __init__(self, secret_key):
        # What to store
        self.sim_gts = []
        self.sim_fused_ests_dec = []
        self.sim_fused_covs_dec = []
        self.sim_fused_ests = []
        self.sim_fused_covs = []
        self.sim_close_ests = []
        self.sim_close_covs = []
        self.sim_errors_fused_dec = []
        self.sim_errors_fused = []
        self.sim_errors_close = []
        self.sk = secret_key
        return
    
    def save_est_and_error(self, enc_inf_scale, enc_inf_vec, enc_inf_mat, plain_inf_vec, plain_inf_mat, close_est, close_cov, gt):
        # Save ground truth
        self.sim_gts.append(gt)

        # Decryption and conversion to normal form
        dec_scale = enc_inf_scale.decrypt(self.sk)
        dec_cov = np.linalg.inv((1.0/dec_scale)*np.array([[x.decrypt(self.sk) for x in row] for row in enc_inf_mat]))
        dec_est = dec_cov@((1.0/dec_scale)*np.array([x.decrypt(self.sk) for x in enc_inf_vec]))

        # Conversion of plaintext to normal form
        plain_cov = np.linalg.inv(plain_inf_mat)
        plain_est = plain_cov@plain_inf_vec

        # Stores estimates and errors at each step as simulation progresses
        self.sim_fused_ests_dec.append(dec_est)
        self.sim_fused_covs_dec.append(dec_cov)
        self.sim_fused_ests.append(plain_est)
        self.sim_fused_covs.append(plain_cov)
        self.sim_close_ests.append(close_est)
        self.sim_close_covs.append(close_cov)
        self.sim_errors_fused_dec.append(np.linalg.norm(dec_est-gt))
        self.sim_errors_fused.append(np.linalg.norm(plain_est-gt))
        self.sim_errors_close.append(np.linalg.norm(close_est-gt))
        return

def plot_sim_layout():
    return

def plot_avg_sim_errors(nav_list):
    mean_errors_fused_dec = np.mean([n.sim_errors_fused_dec for n in nav_list], axis=0)
    mean_errors_fused = np.mean([n.sim_errors_fused for n in nav_list], axis=0)
    mean_errors_close = np.mean([n.sim_errors_close for n in nav_list], axis=0)

    fig = plt.figure()
    fig.set_size_inches(w=3.4, h=3.4)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'Simulation Timesteps $k$')
    ax.set_ylabel(r'Mean Square Error (MSE)')

    ax.plot(range(len(mean_errors_fused_dec)), mean_errors_fused_dec, c='tab:green', label=r'Enc. FCI')
    ax.plot(range(len(mean_errors_fused)), mean_errors_fused, c='tab:blue', linestyle='--', label=r'FCI')
    ax.plot(range(len(mean_errors_close)), mean_errors_close, c='tab:red', linestyle='--', label=r'Nearest')

    ax.legend()

    plt.show()

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

    # Base measurement model for each sensor
    H = np.array([[1, 0, 0, 0], 
                  [0, 0, 1, 0]])
    
    R = np.array([[1, 0], 
                  [0, 1]])

    # Filter init
    init_state = np.array([0, 1, 0, 1])
    init_cov = np.array([[1,   0,  0,   0], 
                         [0, 0.1,  0,   0], 
                         [0,   0,  1,   0], 
                         [0,   0,  0, 0.1]])
    
    # Ground truth init
    mean = np.array([0, 1, 0, 1])
    cov = np.array([[1,   0,  0,   0], 
                    [0, 0.1,  0,   0], 
                    [0,   0,  1,   0], 
                    [0,   0,  0, 0.1]])
    
    # Encryption keys (generated once for speed)
    pk,sk = paillier.generate_paillier_keypair(n_length=512)

    # Result storage
    sim_navigators = []
    
    # Loop sims
    for sim_num in range(SIM_RUNS):

        # Ground truth
        gt_init_state = np.random.multivariate_normal(mean, cov)
        ground_truth = est.GroundTruth(F, Q, gt_init_state)

        # Nodes
        nodes = []
        for i in range(NUM_NODES):

            # "Random" measurement covariance for each sensor
            scale = np.array([[6*np.random.random(), 0],[0, 6*np.random.random()]])
            angle = 2*np.pi
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            sen_R = rotation@scale@rotation.T

            # Create node
            s = est.SensorPure(n, m, H, sen_R)
            f = est.KFilter(n, m, F, Q, H, sen_R, init_state, init_cov)
            nodes.append(SensorNode(i, NODE_POSITIONS[i], s, f, pk))
        
        # Navigator (and results)
        nav = Navigator(sk)
        sim_navigators.append(nav)

        # Temp
        fig = plt.figure()
        fig.set_size_inches(w=7, h=7)
        ax = fig.add_subplot(111)
        
        # Start sim
        print("Running Simulation %d ..." % sim_num)
        for k in range(SIM_STEPS):

            # Each node measures and/or estimates
            gt = ground_truth.update()
            for n in nodes:
                z = n.make_local_est(gt)

                # Temp
                # if k % 2 == 0:
                #    ax.scatter([z[0]], [z[1]], c='black', marker='.')
                #    ax.add_artist(plot.get_cov_ellipse(n.sensor.R, z, 2, fill=False, linestyle='-', edgecolor='black'))
        
            # Each node fuses local with neighbours
            for n in nodes:
                neighbours = [x for i,x in enumerate(nodes) if NODE_CONNECTIONS[n.ident][i] == 1]
                n.fuse_with(neighbours)

            # Navigator takes closest fusion
            closest_node = None
            min_dist = NODE_GRID_DIST*NUM_NODES
            for n in nodes:
                dist = np.linalg.norm(n.position - np.array([gt[0], gt[2]]))
                if dist < min_dist:
                    closest_node = n
            nav.save_est_and_error(closest_node.enc_curr_fuse_scale,
                                   closest_node.enc_curr_fused_inf_vec, 
                                   closest_node.enc_curr_fused_inf_mat, 
                                   closest_node.curr_fused_inf_vec, 
                                   closest_node.curr_fused_inf_mat, 
                                   closest_node.curr_est, 
                                   closest_node.curr_cov, 
                                   gt)
            
            # Temp
            if k % 5 == 0:
                print(k)
            
        # Temp
        # ax.plot([x[0] for x in nav.sim_gts], [x[2] for x in nav.sim_gts], c='gray')
        # ax.plot([x[0] for x in nav.sim_fused_ests_dec], [x[2] for x in nav.sim_fused_ests_dec], c='green', linestyle='-')
        # ax.plot([x[0] for x in nav.sim_fused_ests], [x[2] for x in nav.sim_fused_ests], c='blue', linestyle='--')
        # ax.plot([x[0] for x in nav.sim_close_ests], [x[2] for x in nav.sim_close_ests], c='red', linestyle='--')
        # for k in range(len(nav.sim_gts)):
        #     if k % 2 == 0:
        #         ax.add_artist(plot.get_cov_ellipse(np.array([[nav.sim_fused_covs_dec[k][0][0], nav.sim_fused_covs_dec[k][0][2]],[nav.sim_fused_covs_dec[k][2][0], nav.sim_fused_covs_dec[k][2][2]]]), 
        #                                     np.array([nav.sim_fused_ests_dec[k][0], nav.sim_fused_ests_dec[k][2]]), 
        #                                     2, fill=False, linestyle='-', edgecolor='green'))

        #         ax.add_artist(plot.get_cov_ellipse(np.array([[nav.sim_fused_covs[k][0][0], nav.sim_fused_covs[k][0][2]],[nav.sim_fused_covs[k][2][0], nav.sim_fused_covs[k][2][2]]]), 
        #                                     np.array([nav.sim_fused_ests[k][0], nav.sim_fused_ests[k][2]]), 
        #                                     2, fill=False, linestyle='--', edgecolor='blue'))

        #         ax.add_artist(plot.get_cov_ellipse(np.array([[nav.sim_close_covs[k][0][0], nav.sim_close_covs[k][0][2]],[nav.sim_close_covs[k][2][0], nav.sim_close_covs[k][2][2]]]), 
        #                                     np.array([nav.sim_close_ests[k][0], nav.sim_close_ests[k][2]]), 
        #                                     2, fill=False, linestyle='--', edgecolor='red'))
        # plt.show()

    # Plotting
    plot_avg_sim_errors(sim_navigators)

    return


if __name__ == '__main__':
    main()