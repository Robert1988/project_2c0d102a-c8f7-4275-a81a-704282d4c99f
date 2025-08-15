
import numpy as np
from typing import List, Tuple

class ContextualBrokerage:
    """
    Implementation of the contextual brokerage algorithm from the paper:
    "A Parametric Contextual Online Learning Theory of Brokerage"
    """
    def __init__(self, d: int, L: float, T: int):
        """
        Initialize the contextual brokerage algorithm.
        
        Args:
            d: Dimension of context vectors
            L: Bound on density of valuations
            T: Time horizon
        """
        self.d = d
        self.L = L
        self.T = T
        self.phi_hat = np.zeros(d)  # Current estimate of phi
        self.X = np.zeros((1, d))   # Context matrix
        self.Y = np.zeros(1)        # Feedback vector
        self.explore_count = 0      # Count of exploration steps
        
    def update_estimate(self, c: np.ndarray, y: float):
        """Update the estimate of phi using new context and feedback"""
        self.X = np.vstack([self.X, c])
        self.Y = np.append(self.Y, y)
        A = self.X.T @ self.X + np.eye(self.d)/self.d
        self.phi_hat = np.linalg.inv(A) @ self.X.T @ self.Y
        
    def choose_price(self, c: np.ndarray, t: int) -> float:
        """
        Choose price based on context and current time step
        
        Args:
            c: Current context vector
            t: Current time step
            
        Returns:
            Price to post
        """
        if t == 1:
            return np.random.uniform(0, 1)
            
        # Compute exploration threshold
        A = self.X.T @ self.X + np.eye(self.d)/self.d
        norm = np.sqrt(2 * c.T @ np.linalg.inv(A) @ c)
        threshold = np.sqrt(2 * self.d * np.log(1 + 2 * self.d * (t-1)) / (self.L * self.T))
        
        if norm > threshold:
            # Exploration
            self.explore_count += 1
            return np.random.uniform(0, 1)
        else:
            # Exploitation
            return np.clip(c @ self.phi_hat, 0, 1)

def generate_valuations(m: float, L: float) -> Tuple[float, float]:
    """
    Generate trader valuations as perturbations of market value m
    
    Args:
        m: Market value
        L: Bound on density of valuations
        
    Returns:
        Tuple of (V, W) valuations
    """
    # Generate zero-mean perturbations with bounded density
    xi = np.random.uniform(-1/(2*L), 1/(2*L))
    zeta = np.random.uniform(-1/(2*L), 1/(2*L))
    V = np.clip(m + xi, 0, 1)
    W = np.clip(m + zeta, 0, 1)
    return V, W

def gain_from_trade(p: float, V: float, W: float) -> float:
    """
    Compute gain from trade for given price and valuations
    
    Args:
        p: Posted price
        V: First trader's valuation
        W: Second trader's valuation
        
    Returns:
        Gain from trade
    """
    if min(V, W) <= p <= max(V, W):
        return max(V, W) - min(V, W)
    return 0

def run_experiment(d: int, L: float, T: int) -> dict:
    """
    Run the contextual brokerage experiment
    
    Args:
        d: Dimension of context vectors
        L: Bound on density of valuations
        T: Time horizon
        
    Returns:
        Dictionary containing results
    """
    # True parameter vector (unknown to algorithm)
    phi = np.random.uniform(0, 1, d)
    phi = phi / np.sum(phi)  # Normalize to keep market values in [0,1]
    
    broker = ContextualBrokerage(d, L, T)
    total_gft = 0
    optimal_gft = 0
    
    for t in range(1, T+1):
        # Generate random context
        c = np.random.uniform(0, 1, d)
        c = c / np.sum(c)  # Normalize to keep market values in [0,1]
        
        # Compute market value
        m = c @ phi
        
        # Generate valuations
        V, W = generate_valuations(m, L)
        
        # Choose price
        p = broker.choose_price(c, t)
        
        # Get feedback (2-bit)
        D1 = int(p <= V)
        D2 = int(p <= W)
        
        # Update estimate if in exploration
        if t == 1 or (t > 1 and p != np.clip(c @ broker.phi_hat, 0, 1)):
            broker.update_estimate(c, D1)
        
        # Compute gains
        gft = gain_from_trade(p, V, W)
        optimal_gft += gain_from_trade(m, V, W)
        total_gft += gft
    
    regret = optimal_gft - total_gft
    explore_rate = broker.explore_count / T
    
    return {
        'total_regret': regret,
        'average_regret': regret / T,
        'explore_rate': explore_rate,
        'optimal_gft': optimal_gft,
        'achieved_gft': total_gft
    }

# Experiment parameters
d = 5      # Context dimension
L = 2.0    # Density bound
T = 1000   # Time horizon

# Run experiment
results = run_experiment(d, L, T)

# Print results
print("Experimental Results:")
print(f"Dimension (d): {d}")
print(f"Density bound (L): {L}")
print(f"Time horizon (T): {T}")
print(f"Total regret: {results['total_regret']:.2f}")
print(f"Average regret: {results['average_regret']:.4f}")
print(f"Exploration rate: {results['explore_rate']:.4f}")
print(f"Optimal GFT: {results['optimal_gft']:.2f}")
print(f"Achieved GFT: {results['achieved_gft']:.2f}")
print(f"Regret ratio: {results['total_regret']/results['optimal_gft']:.4f}")

# Theoretical bounds
theoretical_bound = 6 * np.sqrt(L * d * T * np.log(T))
print(f"\nTheoretical regret bound (from paper): {theoretical_bound:.2f}")
print(f"Actual regret is {results['total_regret']/theoretical_bound*100:.1f}% of theoretical bound")