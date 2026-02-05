import numpy as np

class RigorousLinearTiltModel:
    def __init__(self, baseline_pmf_func, basis_matrix):
        self.w_func = baseline_pmf_func
        self.psi = basis_matrix 
        self.K = basis_matrix.shape[1] - 1
        self.theta = np.zeros(self.K)


    """선택적 파라미터(Isolation Property)를 반영한 로그 우도"""
    def get_log_likelihood(self, data, theta_active, active_indices):
        full_theta = np.zeros(self.K)
        for i, idx in enumerate(active_indices):
            full_theta[idx] = theta_active[i]
            
       
        z_vals = 1.0 + np.dot(self.psi[data, 1:], full_theta)
        # 확률이 음수가 되면 절대적인 그 그 값을 극도로 낮은 값으로 반환
        # x값이 커질때 혹시나 음수성을 갖게 될수도 있으므로 pmf를 정의하기위헤 비음수성 정의제약을 걸어둠
        if np.any(z_vals <= 0):
            return -1e15 
            
        return np.sum(np.log(z_vals))

    def pmf(self):
        """전체 그리드에 대한 최종 추정 PMF 계산 (정규화 포함)"""
        x_grid = np.arange(self.psi.shape[0])
        z_vals = 1.0 + np.dot(self.psi[:, 1:], self.theta)
        raw_pmf = self.w_func(x_grid) * z_vals
        
        # 💡 논문급 엄밀성: 유한 그리드에서 합이 1이 되도록 재정규화
        return raw_pmf / np.sum(raw_pmf)