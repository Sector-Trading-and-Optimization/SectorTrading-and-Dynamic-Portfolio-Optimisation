import numpy as np
import cvxpy as cp # to solve optimisation problem
import matplotlib.pyplot as plt
import seaborn as sns

class Optimizer:
    def __init__(self, returns_df):
        self.returns = returns_df
        self.mu = returns_df.mean().values
        self.cov = returns_df.cov().values
        self.n = len(self.mu)

    def mean_variance(self, risk_aversion=7.0):
        w = cp.Variable(self.n)
        ret = w @ self.mu #matmul
        risk = cp.quad_form(w, self.cov)
        prob = cp.Problem(cp.Maximize(ret - risk_aversion * risk), [cp.sum(w) == 1])
        prob.solve()
        return w.value, 'Mean-Variance'