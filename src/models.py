import numpy as np

class RigorousLinearTiltModel:
    def __init__(self, baseline_pmf_func, basis_matrix):
        self.w_func = baseline_pmf_func
        self.psi = basis_matrix # shape: (N, K+1)
        self.K = basis_matrix.shape[1] - 1
        self.theta = np.zeros(self.K)

    def get_log_likelihood(self, data, theta_active, active_indices):
        """선택적 파라미터(Isolation Property)를 반영한 로그 우도"""
        full_theta = np.zeros(self.K)
        for i, idx in enumerate(active_indices):
            full_theta[idx] = theta_active[i]
            
        # Z(x; theta) = 1 + sum theta_k * psi_k(x)
        z_vals = 1.0 + np.dot(self.psi[data, 1:], full_theta)
        
        if np.any(z_vals <= 0):
            return -1e15 # Feasibility 위반 시 극소값 반환
            
        return np.sum(np.log(z_vals))

    def pmf(self):
        """전체 그리드에 대한 최종 추정 PMF 계산 (정규화 포함)"""
        x_grid = np.arange(self.psi.shape[0])
        z_vals = 1.0 + np.dot(self.psi[:, 1:], self.theta)
        raw_pmf = self.w_func(x_grid) * z_vals
        
        # 💡 논문급 엄밀성: 유한 그리드에서 합이 1이 되도록 재정규화
        return raw_pmf / np.sum(raw_pmf)