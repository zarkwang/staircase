import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import optimize

# -----------------------------
#   Weighted Tobit Regression
# -----------------------------
# Tobit model
class tobit_WLS:
    def __init__(self,y,X,weights=None,reg_var_names=None,init_beta=None,init_sigma=1.0):
        self.y = y.values
        self.X = X.values
        
        self.init_sigma = init_sigma

        if weights is None:
            self.weights = np.array([1]*len(y))
        else:
            self.weights = weights.values

        if reg_var_names is None:
            self.reg_var_names = X.columns

        if init_beta is None:
            self.init_beta = np.array([0]*X.shape[1])
        
    # log-likelihood function
    @staticmethod
    def weighted_tobit_loglike(params, X, y, weights):
        beta = params[:-1]  
        sigma = params[-1]  
        y_star = X @ beta 

        ll = np.where(
            y == 0,
            st.norm.logcdf(- y_star / sigma),  # left-truncated
            np.where(
                y == 1,
                st.norm.logcdf((y_star - 1) / sigma),  # right-truncated
                st.norm.logpdf((y - y_star) / sigma) - np.log(sigma)  # uncensored
            )
        )
        return -np.sum(weights * ll)  # objective
    
    def fit(self,robust=False):
        # Initial parameters (beta, sigma)
        initial_params = np.append(self.init_beta, self.init_sigma)
        # Optimization
        self.result = optimize.minimize(self.weighted_tobit_loglike, initial_params, 
                           args=(self.X, self.y, self.weights), 
                           method='L-BFGS-B',
                           bounds = [(None,None)]*self.X.shape[1] + [(0,None)])
        
        # F-test
        # Restricted model
        X_restricted = np.ones((len(self.y), 1))  # Only intercept
        init_params_restricted = np.append(self.init_beta[0], self.init_sigma)  # (intercept, sigma)

        result_restricted = optimize.minimize(self.weighted_tobit_loglike, init_params_restricted, 
                                    args=(X_restricted, self.y, self.weights), 
                                    method='L-BFGS-B',
                                    bounds = [(None,None),(0,None)])

        # Calculate LR statistics
        self.LR = 2 * (result_restricted.fun - self.result.fun)
        self.f_diff_dof = self.X.shape[1] - X_restricted.shape[1] # difference in degree of freedom

        self.f_pvalue = st.chi2.sf(self.LR, self.f_diff_dof)

        #Pseudo R²
        self.pseudo_r2 = 1 - (self.result.fun / result_restricted.fun)

        #AIC
        self.aic = 2 * self.X.shape[1] + 2 * self.result.fun
        
        # t-tests
        # Compute (X'WX)^(-1)
        W = np.diag(self.weights) 
        XTX_inv = np.linalg.inv(self.X.T @ W @ self.X)

        residuals = self.y - self.X @ self.result.x[:-1]

        if not robust:
            # Estimate variance of residuals (σ^2)
            sigma2 = (residuals**2 @ self.weights) / (len(self.y) - self.X.shape[1])  # Weighted MSE

            # Compute standard errors
            self.se = np.sqrt(np.diag(XTX_inv * sigma2))
        else:
            # Compute robust sandwich covariance matrix
            S = np.zeros((self.X.shape[1], self.X.shape[1])) 
            for i in range(len(self.y)):
                xi = self.X[i, :].reshape(-1, 1)  
                S += (residuals[i] ** 2 * self.weights[i]) * (xi @ xi.T)  

            # Compute robust variance-covariance matrix
            robust_cov = XTX_inv @ S @ XTX_inv

            # Compute robust standard errors
            self.se = np.sqrt(np.diag(robust_cov))
        
        # Hessian matrix
        # hessian_inv = tobit_result.hess_inv.todense() 
        # standard_errors = np.sqrt(np.diag(hessian_inv))
       
        self.t = self.result.x[:-1] / self.se
        self.pvalues = [2 * (1 - st.t.cdf(np.abs(t), df=(len(self.y) - self.X.shape[1] - 1))) for t in self.t]

    def result_summary(self):
        # write the t-test results
        t_test_result = {'variable':self.reg_var_names,
                        'coef': self.result.x[:-1],
                        'se': self.se,
                        't': self.t,
                        'p': self.pvalues}
        
        print(f"LR: {self.LR:.4f}, Degree of freedom: {self.f_diff_dof}, F-test pvalue: {self.f_pvalue:.4f}")
        print(f"Pseudo R²: {self.pseudo_r2:.4f}")
        print(f"AIC: {self.aic:.4f}")
        # print(pd.DataFrame(t_test_result))

        return pd.DataFrame(t_test_result)