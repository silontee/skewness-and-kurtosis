import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson, nbinom

def get_rigorous_xmax(params, model_type='Poisson', q=0.999999):
    if model_type == 'Poisson':
        mu = params
        x_max = int(poisson.ppf(q, mu))
        while poisson.cdf(x_max, mu) < q: x_max += 1
        while x_max > 0 and poisson.cdf(x_max-1, mu) >= q: x_max -= 1
    else:
        beta, c = params
        n, p = beta, 1.0 - c
        x_max = int(nbinom.ppf(q, n, p))
        while nbinom.cdf(x_max, n, p) < q: x_max += 1
        while x_max > 0 and nbinom.cdf(x_max-1, n, p) >= q: x_max -= 1
    return x_max

class RigorousOptimizer:
    def __init__(self, model, data, active_indices):
        self.model = model
        self.data = data
        self.active_indices = active_indices

    def optimize(self):
        def constraint(t_active):
            full_theta = np.zeros(self.model.K)
            for i, idx in enumerate(self.active_indices):
                full_theta[idx] = t_active[i]
            return 1.0 + np.dot(self.model.psi[:, 1:], full_theta)

        res = minimize(
            fun=lambda t: -self.model.get_log_likelihood(self.data, t, self.active_indices),
            x0=np.zeros(len(self.active_indices)),
            method='SLSQP',
            constraints=[{'type': 'ineq', 'fun': constraint}],
            options={'ftol': 1e-11}
        )
        final_theta = np.zeros(self.model.K)
        if res.success:
            for i, idx in enumerate(self.active_indices):
                final_theta[idx] = res.x[i]
        self.model.theta = final_theta
        return final_theta