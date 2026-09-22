//! Allocation-free bandit scoring. Engines own legality, perspective and opportunity semantics.

/// Empirical arm statistics on the [-1, 1] utility scale.
#[derive(Clone, Copy, Debug)]
pub struct SelectionStats {
    /// Already oriented: higher is better for the acting player.
    pub mean: f64,
    /// Population variance of the same samples counted by `action_visits`.
    pub variance: f64,
    pub action_visits: u64,
    /// Parent visits in MCTS, action availability in SO-ISMCTS.
    pub opportunities: f64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BanditPolicy {
    #[default]
    Uct,
    Ucb1Tuned,
}

impl BanditPolicy {
    /// `exploration` affects UCT only. Unvisited arms have infinite priority.
    #[inline]
    pub fn score(self, stats: SelectionStats, exploration: f64) -> f64 {
        if stats.action_visits == 0 {
            return f64::INFINITY;
        }
        match self {
            Self::Uct => {
                stats.mean
                    + exploration * (stats.opportunities.ln() / stats.action_visits as f64).sqrt()
            }
            Self::Ucb1Tuned => {
                // Normalize variance to [0,1], then return the bonus to [-1,1].
                let variance = (stats.variance / 4.0).clamp(0.0, 0.25);
                let ratio = stats.opportunities.max(1.0).ln() / stats.action_visits as f64;
                let bound = (variance + (2.0 * ratio).sqrt()).min(0.25);
                stats.mean + 2.0 * (ratio * bound).sqrt()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uct_uses_supplied_opportunities() {
        for opportunities in [100.0_f64, 25.0] {
            let stats = SelectionStats {
                mean: 0.4,
                variance: 0.0,
                action_visits: 10,
                opportunities,
            };
            assert_eq!(
                BanditPolicy::Uct.score(stats, 1.0),
                0.4 + (opportunities.ln() / 10.0).sqrt()
            );
        }
    }
    #[test]
    fn tuned_normalization_and_orientation() {
        for mean in [0.25, -0.25] {
            // Utilities [1,-1,1,0]: sum=1, squared sum=3, variance=11/16.
            let stats = SelectionStats {
                mean,
                variance: 11.0 / 16.0,
                action_visits: 4,
                opportunities: 25.0,
            };
            let ratio = 25.0_f64.ln() / 4.0;
            let expected =
                mean + 2.0 * (ratio * (11.0 / 64.0 + (2.0 * ratio).sqrt()).min(0.25)).sqrt();
            assert_eq!(BanditPolicy::Ucb1Tuned.score(stats, 0.0), expected);
            assert_eq!(BanditPolicy::Ucb1Tuned.score(stats, 99.0), expected);
        }
    }
    #[test]
    fn unvisited_priority_avoids_zero_division() {
        let stats = SelectionStats {
            mean: 0.0,
            variance: 0.0,
            action_visits: 0,
            opportunities: 0.0,
        };
        for policy in [BanditPolicy::Uct, BanditPolicy::Ucb1Tuned] {
            assert_eq!(policy.score(stats, 0.0), f64::INFINITY);
        }
    }

    #[test]
    fn tuned_uses_variance_below_the_cap() {
        let stats = SelectionStats {
            mean: 0.4,
            variance: 0.01,
            action_visits: 10_000,
            opportunities: 10_000.0,
        };
        let ratio = stats.opportunities.ln() / stats.action_visits as f64;
        let bound = 0.01 / 4.0 + (2.0 * ratio).sqrt();
        assert!(bound < 0.25);
        assert_eq!(
            BanditPolicy::Ucb1Tuned.score(stats, 1.0),
            0.4 + 2.0 * (ratio * bound).sqrt()
        );
        assert!(
            BanditPolicy::Ucb1Tuned.score(
                SelectionStats {
                    variance: 0.2,
                    ..stats
                },
                1.0
            ) > BanditPolicy::Ucb1Tuned.score(stats, 1.0)
        );
    }
}
