import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
class Optimizer:
    def __init__(self, returns_df):
        self.returns = returns_df
        self.mu = returns_df.mean().values  # Expected returns
        self.cov = returns_df.cov().values  # Covariance matrix
        self.n = len(self.mu)  # Number of assets

    def mean_variance(self, risk_aversion=1.0):
        w = cp.Variable(self.n)
        ret = w @ self.mu
        risk = cp.quad_form(w, self.cov)
        prob = cp.Problem(cp.Maximize(ret - risk_aversion * risk), [cp.sum(w) == 1, w >= 0])
        prob.solve()
        return w.value, 'Mean-Variance'

    def min_variance(self):
        w = cp.Variable(self.n)
        risk = cp.quad_form(w, self.cov)
        prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w >= 0])
        prob.solve()
        return w.value, 'Minimum-Variance'

    def risk_parity(self, iterations=1000, tol=1e-6):
        w = np.ones(self.n) / self.n
        for _ in range(iterations):
            risk_contrib = w * (self.cov @ w)
            total_risk = np.dot(w, self.cov @ w)
            diff = risk_contrib - total_risk / self.n
            if np.linalg.norm(diff) < tol:
                break
            w *= 1 - 0.1 * diff / risk_contrib
            w = np.maximum(w, 0)
            w /= np.sum(w)
        return w, 'Risk-Parity'

    def plot_portfolio_comparison(self):
      strategies = ['Mean-Variance', 'Minimum-Variance', 'Risk-Parity']
      weights_list = []
      returns_annual = []
      risks_annual = []
      sharpe_ratios = []

      # Calculate metrics for each strategy
      w_mv, _ = self.mean_variance()
      w_min, _ = self.min_variance()
      w_rp, _ = self.risk_parity()

      for w, name in [(w_mv, 'Mean-Variance'), (w_min, 'Minimum-Variance'), (w_rp, 'Risk-Parity')]:
          if w is not None:
              weights_list.append(w)
              port_return = w @ self.mu * 252
              port_risk = np.sqrt(w @ self.cov @ w) * np.sqrt(252)
              sharpe = port_return / port_risk if port_risk > 0 else 0

              returns_annual.append(port_return)
              risks_annual.append(port_risk)
              sharpe_ratios.append(sharpe)
          else:
              weights_list.append(np.ones(self.n) / self.n)
              returns_annual.append(0)
              risks_annual.append(0)
              sharpe_ratios.append(0)

      # Plot only the bottom two: Sharpe Ratio + Correlation Matrix
      fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns

      # Sharpe ratio comparison
      colors = ['blue', 'red', 'green']
      bars = axes[0].bar(strategies, sharpe_ratios, color=colors, alpha=0.7)
      axes[0].set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
      axes[0].set_ylabel('Sharpe Ratio')
      axes[0].grid(True, alpha=0.3)

      for bar, value in zip(bars, sharpe_ratios):
          axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom')

      # Correlation matrix
      corr_matrix = self.returns.corr()
      sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                  square=True, ax=axes[1])
      axes[1].set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')

      plt.tight_layout()
      plt.show()

      return strategies, weights_list, returns_annual, risks_annual, sharpe_ratios
    
def main():

    try:
        # Portfolio optimization
        print_section_header("Portfolio Optimization")
        optimizer = Optimizer(returns_all)

        # Compare different optimization strategies
        methods, weights_list, returns_annual, risks_annual, sharpe_ratios = optimizer.plot_portfolio_comparison()

        # Select the best strategy based on Sharpe ratio
        best_idx = sharpe_ratios.index(max(sharpe_ratios))
        best_method = methods[best_idx]
        best_weights = weights_list[best_idx]
        best_return = returns_annual[best_idx]
        best_risk = risks_annual[best_idx]
        best_sharpe = sharpe_ratios[best_idx]

        # Create weights dictionary
        weights_dict = dict(zip(selected_tickers, best_weights))

        print(f"\n🎯 Selected Strategy: {best_method}")
        print(f"📊 Expected Annual Return: {best_return:.2%}")
        print(f"📊 Expected Annual Risk: {best_risk:.2%}")
        print(f"📊 Expected Sharpe Ratio: {best_sharpe:.3f}")
        print("\n💼 Optimal Portfolio Weights:")
        for ticker, weight in weights_dict.items():
            print(f"  {ticker}: {weight:.2%}")

    except Exception as e:
        print(f"\n❌ Critical Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()