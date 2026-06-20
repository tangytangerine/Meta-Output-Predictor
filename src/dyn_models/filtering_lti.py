import numpy as np
import scipy as sp
import sympy
from filterpy.kalman import KalmanFilter

def softplus(x):
    return np.log(1 + np.exp(x))

class FilterSim:
    def __init__(self, nx=3, ny=2, sigma_w=1e-1, sigma_v=1e-1, tri=False, n_noise=1):
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v

        self.n_noise = n_noise
        
        def random_nonsymmetric_with_eigs(n, eig_low, eig_high, cond_max=1e3, seed=None, max_tries=10000):
            rng = np.random.default_rng(seed)

            eigs = rng.uniform(eig_low, eig_high, size=n)
            Lambda = np.diag(eigs)

            for _ in range(max_tries):
                P = rng.normal(size=(n, n))
                if np.linalg.matrix_rank(P) < n:
                    continue

                c = np.linalg.cond(P)
                if c <= cond_max:
                    A = P @ Lambda @ np.linalg.inv(P)
                    return A

            raise RuntimeError("Could not find a well-conditioned P within max_tries")
        
        gen = np.random.default_rng()
        lims = np.array([0.5, 0.6])

        if tri:
            A = np.diag(gen.uniform(lims[0], lims[-1], (nx)))
            A[np.triu_indices(nx,1)] = gen.uniform(lims[0], lims[-1], (nx**2+nx)//2-nx)
            self.A = A
        else:
            # A = np.random.rand(nx, nx)
            A = random_nonsymmetric_with_eigs(nx, lims[0], lims[-1])
            self.A = A

        self.C = np.eye(nx) if nx == ny else self.construct_C(self.A, ny)
    
    def simulate(self, traj_len, x0=None):
        ny, nx = self.C.shape
        n_noise = self.n_noise
        mean_v, mean_w = np.random.normal(0,1,ny), np.random.normal(0,1,nx)
        mean_v, mean_w = 0, 0
        xs = [np.random.randn(nx) if x0 is None else x0]
        xs_cl = [np.random.randn(nx) if x0 is None else x0]
        vs = [(np.random.randn(ny) + mean_v) * self.sigma_v for _ in range(n_noise)]
        ws = [(np.random.randn(nx) + mean_w) * self.sigma_w for _ in range(n_noise)]
        ys = [self.C @ xs[0]+ sum(vs)]
        ys_cl = [self.C @ xs_cl[0]+ sum(vs)]
        
        ulims = np.array([-1, 1])
        us = []
        for _ in range(traj_len):
            u = np.random.uniform(ulims[0], ulims[1], size=nx)
            # u = np.zeros((nx))
            us.append(u)
            
            x_cl = self.A @ xs_cl[-1] + sum(ws[-n_noise:])
            x = self.A @ xs[-1] + sum(ws[-n_noise:]) + u
            
            xs_cl.append(x_cl)
            xs.append(x)
            ws.append((np.random.randn(nx) + mean_w) * self.sigma_w)
            
            vs.append((np.random.randn(ny) + mean_v) * self.sigma_v)
            y_cl = self.C @ xs_cl[-1] + sum(vs[-n_noise:])
            y = self.C @ xs[-1] + sum(vs[-n_noise:])
            ys_cl.append(y_cl)
            ys.append(y)
        return np.array(xs).astype("f"), np.array(ys).astype("f"), np.array(us).astype("f"), np.array(xs_cl).astype("f"), np.array(ys_cl).astype("f")

    @staticmethod
    def construct_C(A, ny):
        nx = A.shape[0]
        _O = [np.eye(nx)]
        for _ in range(nx-1):
            _O.append(_O[-1]@A)
        while True:
            C = np.random.rand(ny, nx) 
            O = np.concatenate([C@o for o in _O], axis=0)
            if np.linalg.matrix_rank(O) == nx:
                break
        return C.astype("f")

def apply_kf(fsim, ys, x0=None, P0=None, sigma_w=None, sigma_v=None, return_obj=False):
    ny, nx = fsim.C.shape

    sigma_w = fsim.sigma_w if sigma_w is None else sigma_w
    sigma_v = fsim.sigma_v if sigma_v is None else sigma_v

    f = KalmanFilter(dim_x=nx, dim_z=ny)
    f.Q = np.eye(nx) * sigma_w ** 2
    f.R = np.eye(ny) * sigma_v ** 2
    f.P = np.eye(nx) if P0 is None else P0
    f.x = np.zeros(nx) if x0 is None else x0
    f.F = fsim.A
    f.H = fsim.C

    ls = [fsim.C @ f.x]
    for y in ys:
        f.update(y)
        f.predict()
        ls.append(fsim.C @ f.x)
    ls = np.array(ls)
    return (f,ls) if return_obj else ls
    
def _generate_lti_sample(dataset_typ, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    fsim = FilterSim(nx, ny, sigma_w, sigma_v, tri="upperTriA" == dataset_typ, n_noise=n_noise)
    states, obs, us, states_cl, obs_cl = fsim.simulate(n_positions)
    return fsim, {"states": states, "inputs" : us, "obs": obs, "statesCL": states_cl, "obsCL": obs_cl, "A": fsim.A, "C": fsim.C}
    
def generate_lti_sample(dataset_typ, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    while True:
        fsim, entry = _generate_lti_sample(dataset_typ, n_positions, nx, ny, sigma_w, sigma_v, n_noise=n_noise)
        if check_validity(entry):
            return fsim, entry
        
def generate_changing_lti_sample(n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    fsim1 = FilterSim(nx=nx, ny=ny, sigma_w=sigma_w, sigma_v=sigma_v, n_noise=n_noise)
    fsim2 = FilterSim(nx=nx, ny=ny, sigma_w=sigma_w, sigma_v=sigma_v, n_noise=n_noise)

    _xs, _ys = fsim1.simulate(n_positions)
    while not check_validity({"states":_xs, "obs":_ys}):
        _xs, _ys = fsim1.simulate(n_positions)
    _xs_cont, _ys_cont = fsim2.simulate(n_positions, x0=_xs[-1])
    while not check_validity({"states":_xs_cont, "obs":_ys_cont}):
        _xs_cont, _ys_cont = fsim2.simulate(n_positions, x0=_xs[-1])
    y_seq = np.concatenate([_ys[:-1], _ys_cont], axis=0)
    return fsim1, {"obs": y_seq}
    
def check_validity(entry):
    if entry is None:
        return False
    states, obs = entry["states"], entry["obs"]
    return np.max(np.abs(states)) < 50 and np.max(np.abs(obs)) < 50


