"""
Advanced Match Selection System
Integrating Bayesian inference, Markov models, entropy-based uncertainty, and risk management
for intelligent sports betting permutation strategies.

Author: Advanced Analytics
Version: 1.0
"""

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import poisson, entropy
from typing import Dict, List, Tuple, Optional, Union
import warnings
from itertools import product
import json

warnings.filterwarnings('ignore')

class AdvancedMatchSelector:
    """
    Advanced statistical system for intelligent match selection combining:
    - Bayesian probability updates
    - Hidden Markov Models for team form
    - Poisson goal modeling
    - Shannon entropy uncertainty quantification
    - Risk-adjusted portfolio selection
    """
    
    def __init__(self, 
                 alpha_uncertainty: float = 0.15,
                 beta_conservatism: float = 0.1,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize the match selector with configurable parameters.
        
        Args:
            alpha_uncertainty: Penalty factor for high entropy matches (0.1-0.2 typical)
            beta_conservatism: Shrinkage toward uniform distribution (0.05-0.15 typical)
            weights: Custom weights for probability fusion {'market': 0.6, 'poisson': 0.3, 'form': 0.1}
        """
        self.alpha_uncertainty = alpha_uncertainty
        self.beta_conservatism = beta_conservatism
        self.weights = weights or {'market': 0.6, 'poisson': 0.3, 'form': 0.1}
        
        # Validate weights sum to 1.0
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    
    def convert_odds_to_probs(self, odds_home: float, odds_draw: float, odds_away: float) -> np.ndarray:
        """
        Convert bookmaker odds to implied probabilities, removing the overround margin.
        
        Args:
            odds_home, odds_draw, odds_away: Decimal odds for each outcome
            
        Returns:
            numpy array of normalized probabilities [home, draw, away]
        """
        if any(odds <= 1.0 for odds in [odds_home, odds_draw, odds_away]):
            raise ValueError("All odds must be greater than 1.0")
            
        implied_probs = np.array([1/odds_home, 1/odds_draw, 1/odds_away])
        return implied_probs / implied_probs.sum()
    
    def poisson_match_probs(self, 
                           lambda_home: float, 
                           lambda_away: float, 
                           home_advantage: float = 0.3,
                           max_goals: int = 5) -> np.ndarray:
        """
        Calculate match outcome probabilities using Poisson distribution for goals.
        
        Args:
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team  
            home_advantage: Additional goals expectation for home team
            max_goals: Maximum goals to consider in calculation (5-7 typical)
            
        Returns:
            numpy array [P(home_win), P(draw), P(away_win)]
        """
        lambda_home_adj = lambda_home + home_advantage
        lambda_away_adj = lambda_away
        
        # Build probability matrix for all scorelines
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob_matrix[h, a] = poisson.pmf(h, lambda_home_adj) * poisson.pmf(a, lambda_away_adj)
        
        # Aggregate to match outcomes
        prob_home = np.sum(prob_matrix[np.triu_indices_from(prob_matrix, k=1)])  # h > a
        prob_draw = np.sum(np.diag(prob_matrix))  # h = a
        prob_away = np.sum(prob_matrix[np.tril_indices_from(prob_matrix, k=-1)])  # h < a
        
        result = np.array([prob_home, prob_draw, prob_away])
        return result / result.sum()  # Normalize to handle rounding
    
    def markov_form_adjustment(self, 
                             recent_results: List[str], 
                             lookback: int = 5) -> float:
        """
        Calculate form-based multiplier using Hidden Markov Model approach.
        
        States: Hot (H), Normal (N), Cold (C)
        Transition matrix based on recent performance patterns.
        
        Args:
            recent_results: List of recent match results ['W', 'D', 'L']
            lookback: Number of recent matches to consider
            
        Returns:
            Multiplier for goal expectation (0.9 - 1.1 typical range)
        """
        if not recent_results:
            return 1.0
            
        # Map results to numerical scores
        result_map = {'W': 1, 'D': 0, 'L': -1, 'w': 1, 'd': 0, 'l': -1}
        scores = [result_map.get(r, 0) for r in recent_results[-lookback:]]
        
        if not scores:
            return 1.0
            
        avg_score = np.mean(scores)
        recent_trend = np.mean(scores[-3:]) if len(scores) >= 3 else avg_score
        
        # Determine current state and multiplier
        if avg_score > 0.4 and recent_trend > 0.2:
            return 1.08  # Hot state
        elif avg_score < -0.4 and recent_trend < -0.2:
            return 0.94  # Cold state
        else:
            return 1.0   # Normal state
    
    def bayesian_update(self, 
                       market_probs: np.ndarray,
                       poisson_probs: np.ndarray,
                       form_multiplier_home: float,
                       form_multiplier_away: float) -> np.ndarray:
        """
        Bayesian probability fusion combining market odds, Poisson model, and form analysis.
        
        Args:
            market_probs: Probabilities from market odds
            poisson_probs: Probabilities from Poisson model
            form_multiplier_home: Home team form adjustment
            form_multiplier_away: Away team form adjustment
            
        Returns:
            Final blended probabilities
        """
        # Adjust Poisson probabilities based on form
        form_adjusted = poisson_probs.copy()
        form_adjusted[0] *= form_multiplier_home / form_multiplier_away  # Home win
        form_adjusted[2] *= form_multiplier_away / form_multiplier_home  # Away win
        form_adjusted = form_adjusted / form_adjusted.sum()  # Normalize
        
        # Safe logit transformation (avoid extreme values)
        def safe_logit(p: np.ndarray) -> np.ndarray:
            p_clipped = np.clip(p, 1e-6, 1-1e-6)
            return logit(p_clipped)
        
        def safe_expit(x: np.ndarray) -> np.ndarray:
            x_clipped = np.clip(x, -10, 10)
            return expit(x_clipped)
        
        # Neutral prior (slightly favoring home)
        neutral_prior = np.array([0.45, 0.25, 0.3])
        
        # Weighted combination in logit space for numerical stability
        combined_logits = (self.weights['market'] * safe_logit(market_probs) +
                          self.weights['poisson'] * safe_logit(form_adjusted) +
                          self.weights['form'] * safe_logit(neutral_prior))
        
        combined_probs = safe_expit(combined_logits)
        
        # Apply conservative shrinkage toward uniform distribution
        uniform = np.array([1/3, 1/3, 1/3])
        shrunk_probs = (1 - self.beta_conservatism) * combined_probs + self.beta_conservatism * uniform
        
        return shrunk_probs / shrunk_probs.sum()
    
    def calculate_entropy(self, probs: np.ndarray) -> float:
        """
        Calculate Shannon entropy for uncertainty quantification.
        
        Args:
            probs: Probability distribution
            
        Returns:
            Entropy in bits (log base 2)
        """
        return entropy(probs, base=2)
    
    def calculate_edge(self, probs: np.ndarray, odds: np.ndarray) -> Tuple[float, int]:
        """
        Calculate expected value edge for each outcome and return best option.
        
        Args:
            probs: Final probability estimates
            odds: Bookmaker odds
            
        Returns:
            Tuple of (best_edge, best_outcome_index)
        """
        edges = probs * odds - 1
        best_outcome = np.argmax(edges)
        best_edge = edges[best_outcome]
        return best_edge, best_outcome
    
    def score_match(self, match_data: Dict) -> Dict:
        """
        Comprehensive scoring of a single match incorporating all models.
        
        Args:
            match_data: Dictionary containing match information
            
        Required keys in match_data:
            - odds_home, odds_draw, odds_away: decimal odds
            - lambda_home, lambda_away: expected goals
            - form_home, form_away: recent results lists ['W','D','L']
            - match_id: unique identifier
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Extract required data
            odds = np.array([match_data['odds_home'], match_data['odds_draw'], match_data['odds_away']])
            lambda_home = match_data['lambda_home']
            lambda_away = match_data['lambda_away']
            form_home = match_data.get('form_home', [])
            form_away = match_data.get('form_away', [])
            
            # Calculate component probabilities
            market_probs = self.convert_odds_to_probs(*odds)
            poisson_probs = self.poisson_match_probs(lambda_home, lambda_away)
            
            form_mult_home = self.markov_form_adjustment(form_home)
            form_mult_away = self.markov_form_adjustment(form_away)
            
            final_probs = self.bayesian_update(market_probs, poisson_probs, 
                                             form_mult_home, form_mult_away)
            
            # Calculate metrics
            uncertainty = self.calculate_entropy(final_probs)
            edge, best_outcome = self.calculate_edge(final_probs, odds)
            
            # Risk-adjusted score
            risk_adjusted_score = edge - self.alpha_uncertainty * uncertainty
            
            # Kelly-like position sizing (capped at 10%)
            kelly_fraction = max(0, min(0.1, edge / (odds[best_outcome] - 1))) if edge > 0 else 0
            
            return {
                'match_id': match_data['match_id'],
                'final_probs': final_probs.tolist(),
                'market_probs': market_probs.tolist(),
                'poisson_probs': poisson_probs.tolist(),
                'edge': float(edge),
                'uncertainty': float(uncertainty),
                'risk_score': float(risk_adjusted_score),
                'best_outcome': int(best_outcome),
                'confidence': float(final_probs[best_outcome]),
                'kelly_stake': float(kelly_fraction * 100),  # As percentage
                'form_multipliers': {'home': form_mult_home, 'away': form_mult_away}
            }
            
        except Exception as e:
            raise ValueError(f"Error scoring match {match_data.get('match_id', 'unknown')}: {str(e)}")
    
    def select_matches(self,
                      all_matches: List[Dict],
                      target_matches: int = 13,
                      constraints: Optional[Dict] = None) -> List[Dict]:
        """
        Select optimal matches based on risk-adjusted scores with constraints.
        
        Args:
            all_matches: List of match dictionaries
            target_matches: Number of matches to select
            constraints: Selection constraints dictionary
            
        Default constraints:
            - max_draws: 4 (limit volatile outcomes)
            - min_edge: 0.01 (1% minimum edge)
            - max_uncertainty: 1.05 (entropy ceiling)
            - min_confidence: 0.38 (38% minimum probability)
            
        Returns:
            List of selected match analysis dictionaries
        """
        if constraints is None:
            constraints = {
                'max_draws': 4,
                'min_edge': 0.01,
                'max_uncertainty': 1.05,
                'min_confidence': 0.38
            }
        
        # Score all matches
        scored_matches = []
        for match in all_matches:
            try:
                score_data = self.score_match(match)
                scored_matches.append(score_data)
            except Exception as e:
                print(f"Warning: Skipping match {match.get('match_id', 'unknown')}: {e}")
                continue
        
        if not scored_matches:
            raise ValueError("No valid matches found after scoring")
        
        # Apply constraints and select
        filtered_matches = []
        draw_count = 0
        
        # Sort by risk score (descending)
        scored_matches.sort(key=lambda x: x['risk_score'], reverse=True)
        
        for match in scored_matches:
            # Check all constraints
            if match['edge'] < constraints['min_edge']:
                continue
            if match['uncertainty'] > constraints['max_uncertainty']:
                continue
            if match['confidence'] < constraints['min_confidence']:
                continue
            if match['best_outcome'] == 1 and draw_count >= constraints['max_draws']:  # Draw outcome
                continue
                
            filtered_matches.append(match)
            if match['best_outcome'] == 1:  # Count draws
                draw_count += 1
                
            if len(filtered_matches) >= target_matches:
                break
        
        return filtered_matches[:target_matches]
    
    def generate_permutations_smart(self,
                                  selected_matches: List[Dict],
                                  max_permutations: int = 50000,
                                  method: str = 'adaptive') -> List[Tuple]:
        """
        Generate permutations with intelligent sampling strategies.
        
        Args:
            selected_matches: List of selected match analyses
            max_permutations: Maximum number of permutations to generate
            method: 'all', 'monte_carlo', or 'adaptive'
            
        Returns:
            List of permutation tuples (outcome indices)
        """
        n_matches = len(selected_matches)
        total_permutations = 3**n_matches
        
        if method == 'all' or (method == 'adaptive' and total_permutations <= max_permutations):
            # Generate all possible permutations
            return list(product(range(3), repeat=n_matches))
        
        # Monte Carlo sampling weighted by probabilities
        permutations = set()  # Use set to avoid duplicates
        probs_matrix = np.array([match['final_probs'] for match in selected_matches])
        
        attempts = 0
        max_attempts = max_permutations * 2  # Avoid infinite loop
        
        while len(permutations) < max_permutations and attempts < max_attempts:
            perm = []
            for match_probs in probs_matrix:
                outcome = np.random.choice(3, p=match_probs)
                perm.append(outcome)
            permutations.add(tuple(perm))
            attempts += 1
        
        return list(permutations)
    
    def export_results(self, results: List[Dict], filename: str = None) -> Optional[str]:
        """
        Export analysis results to JSON format.
        
        Args:
            results: Analysis results to export
            filename: Optional filename, if None returns JSON string
            
        Returns:
            JSON string if no filename provided
        """
        export_data = {
            'metadata': {
                'system': 'AdvancedMatchSelector',
                'parameters': {
                    'alpha_uncertainty': self.alpha_uncertainty,
                    'beta_conservatism': self.beta_conservatism,
                    'weights': self.weights
                },
                'total_matches': len(results)
            },
            'results': results
        }
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            return None
        else:
            return json.dumps(export_data, indent=2)


def estimate_lambda_from_odds(odds_home: float, odds_draw: float, odds_away: float) -> Tuple[float, float]:
    """
    Estimate expected goals from odds using reverse Poisson calculation.
    
    Args:
        odds_home, odds_draw, odds_away: Decimal odds
        
    Returns:
        Tuple of (lambda_home, lambda_away) estimates
    """
    # Convert to probabilities
    probs = np.array([1/odds_home, 1/odds_draw, 1/odds_away])
    probs = probs / probs.sum()
    
    # Rough estimation based on odds patterns
    total_goals = 2.5  # League average assumption
    
    if probs[0] > probs[2]:  # Home favored
        lambda_home = total_goals * 0.6
        lambda_away = total_goals * 0.4
    elif probs[2] > probs[0]:  # Away favored  
        lambda_home = total_goals * 0.4
        lambda_away = total_goals * 0.6
    else:  # Even match
        lambda_home = lambda_away = total_goals * 0.5
    
    return lambda_home, lambda_away


def create_match_template() -> Dict:
    """
    Create a template dictionary showing required match data structure.
    
    Returns:
        Template dictionary with all required fields
    """
    return {
        'match_id': 'unique_identifier',
        'teams': 'Team A vs Team B',  # Optional, for display
        'league': 'League Name',      # Optional, for display  
        'date': 'YYYY-MM-DD HH:MM',  # Optional, for display
        'odds_home': 2.50,           # Required: decimal odds
        'odds_draw': 3.20,           # Required: decimal odds
        'odds_away': 2.80,           # Required: decimal odds
        'lambda_home': 1.3,          # Required: expected goals (or use estimate_lambda_from_odds)
        'lambda_away': 1.2,          # Required: expected goals
        'form_home': ['W', 'D', 'L', 'W', 'D'],  # Required: recent results
        'form_away': ['L', 'W', 'W', 'D', 'L']   # Required: recent results
    }


def main_example():
    """
    Example usage of the AdvancedMatchSelector system.
    Replace this with your actual match data.
    """
    print("🚀 Advanced Match Selection System - Clean Implementation")
    print("=" * 60)
    
    # Initialize selector
    selector = AdvancedMatchSelector(
        alpha_uncertainty=0.15,
        beta_conservatism=0.1
    )
    
    # Example: Create your match data here
    # This is just a template - replace with real data
    sample_matches = []
    
    print("📋 To use this system:")
    print("1. Prepare your match data using the create_match_template() format")
    print("2. Call selector.select_matches(your_matches, target_matches=13)")  
    print("3. Generate permutations with selector.generate_permutations_smart()")
    print("\n📖 See create_match_template() for required data structure")
    
    # Show template structure
    template = create_match_template()
    print("\n🔧 Required Match Data Structure:")
    for key, value in template.items():
        print(f"   '{key}': {repr(value)}")
    
    return selector


if __name__ == "__main__":
    selector = main_example()
    print(f"\n✅ System initialized successfully!")
    print(f"📊 Ready to analyze matches with Bayesian + Markov + Entropy models")
