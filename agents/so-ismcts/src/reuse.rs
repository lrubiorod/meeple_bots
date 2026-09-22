
#[cfg(test)]
mod tests {
    use super::*;
    use meeple_bots_lost_cities::{LostCities, LostCitiesAction as A, LostCitiesObservation};
    use meeple_bots_simulation::SplitMix64;
    use std::num::NonZeroU32;
    fn ready() -> meeple_bots_lost_cities::LostCitiesState { super::super::tests::ready() }
    fn wrapper(policy: BanditPolicy) -> ReusableSoIsmcts<LostCities, LostCitiesObservation> {
        ReusableSoIsmcts::new(SoIsmctsConfig { tree_reuse: true, selection_policy: policy,
            budget: SearchBudget::Iterations(NonZeroU32::new(128).unwrap()), ..Default::default() })
    }
    #[test]
    fn own_microaction_preserves_subtree_and_adds_only_new_work() {
        for policy in [BanditPolicy::Uct, BanditPolicy::Ucb1Tuned] {
            let mut w = wrapper(policy);
            let mut state = ready();
            let owner = PlayerId::FIRST;
            w.start_match(&LostCities, owner);
            let obs = LostCities.observation(&state, owner);
            let legal: Vec<_> = LostCities.legal_actions(&state).collect();
            let r = w.search(&LostCities, &obs, owner, &legal, &mut SplitMix64::new(1)).unwrap();
            let action = r.search.action;
            let child = r.search.nodes[0].edges.iter().find(|e| e.action == action).unwrap().outcomes[0].1;
            let expected = r.search.nodes[child].clone();
            let old_nodes = r.search.nodes.len();
            LostCities.apply_action(&mut state, &action).unwrap();
            assert_eq!(LostCities.status(&state), PositionStatus::PlayerTurn(owner));
            let obs = LostCities.observation(&state, owner);
            w.advance_real_transition(&LostCities, owner, owner, &action, &obs);
            assert_eq!(w.pending.own_action_hits, 1);
            let retained = &w.retained.as_ref().unwrap().search.nodes;
            assert_eq!(retained[0].visits, expected.visits);
            for (a,b) in retained[0].edges.iter().zip(&expected.edges) {
                assert_eq!((a.visits,a.availability,a.total_utility,a.total_squared_utility),
                           (b.visits,b.availability,b.total_utility,b.total_squared_utility));
            }
            assert_eq!(w.pending.pruned_nodes as usize + retained.len(), old_nodes);
            let legal: Vec<_> = LostCities.legal_actions(&state).collect();
            let r = w.search(&LostCities, &obs, owner, &legal, &mut SplitMix64::new(2)).unwrap();
            assert_eq!(r.search.diagnostics.completed_iterations,128);
            assert_eq!(r.search.diagnostics.determinizations_sampled,128);
            assert_eq!(r.search.nodes[0].visits,expected.visits+128);
            assert_eq!(r.reuse.reused_root_visits as u64,expected.visits);
            assert_eq!(r.new_nodes + r.reuse.reused_nodes,r.search.nodes.len() as u64);
            for n in &r.search.nodes { for e in &n.edges {
                assert!(e.visits <= e.availability);
                assert!(e.outcomes.iter().all(|(_, child)| *child < r.search.nodes.len()));
            }}
        }
    }
    fn fixture() -> ReusableSoIsmcts<LostCities, LostCitiesObservation> {
        let mut w=wrapper(BanditPolicy::Uct);let obs=LostCities.observation(&ready(),PlayerId::FIRST);
        w.start_match(&LostCities,PlayerId::FIRST);w.observation=Some(obs.clone());
        let mut other=obs.clone();other.deck_size-=1;
        let node = |visits| Node {visits,edges:vec![]};
        w.retained=Some(ReuseSearchResult{new_nodes:4,reuse:TreeReuseStats::default(),search:SearchResult{
            action:A::DrawDeck,diagnostics:Diagnostics::default(),nodes:vec![
                Node {visits:12,edges:vec![Edge {action:A::DrawDeck,visits:12,availability:12,total_utility:2.,total_squared_utility:8.,
                    outcomes:vec![(obs,1),(other,2)]}]},node(5),node(7)]}});
        w
    }
    #[test]
    fn outcomes_are_matched_only_beneath_actual_action() {
        let mut w=fixture();let obs=w.retained.as_ref().unwrap().search.nodes[0].edges[0].outcomes[1].0.clone();
        w.advance_real_transition(&LostCities,PlayerId::FIRST,PlayerId::SECOND,&A::DrawDeck,&obs);
        assert_eq!(w.retained.as_ref().unwrap().search.nodes[0].visits,7);
        assert_eq!(w.pending.opponent_action_hits,1);
        assert_eq!(w.pending.pruned_nodes,2);
        let mut w=fixture();let mut obs=obs;obs.deck_size-=1;
        w.advance_real_transition(&LostCities,PlayerId::FIRST,PlayerId::FIRST,&A::DrawDeck,&obs);
        assert!(w.retained.is_none());assert_eq!(w.pending.transition_misses,1);
        let mut w=fixture();let obs=w.observation.clone().unwrap();
        let action=LostCities.legal_actions(&ready()).next().unwrap();
        w.advance_real_transition(&LostCities,PlayerId::FIRST,PlayerId::FIRST,&action,&obs);
        assert!(w.retained.is_none()); // Matching observation under another action is not enough.
    }
    #[test]
    fn owner_observation_lifecycle_and_clone_reset() {
        let mut w=fixture();assert!(w.clone().retained.is_none());assert!(w.clone().owner.is_none());
        let obs=w.observation.clone().unwrap();
        w.advance_real_transition(&LostCities,PlayerId::SECOND,PlayerId::FIRST,&A::DrawDeck,&obs);
        assert!(w.retained.is_none());
        let mut w=fixture();w.start_match(&LostCities,PlayerId::FIRST);assert!(w.retained.is_none());
        let mut w=fixture();w.end_match();assert!(w.retained.is_none());assert!(w.owner.is_none());
        let mut w=fixture();let state=ready();let mut obs=w.observation.clone().unwrap();obs.deck_size-=1;
        // Reset before attempting to sample an invalid observation.
        let _=w.search(&LostCities,&obs,PlayerId::FIRST,&LostCities.legal_actions(&state).collect::<Vec<_>>(),&mut SplitMix64::new(1));
        assert!(w.retained.is_none());
    }
    #[test]
    fn malformed_outcomes_reset_and_no_unreachable_data_survives() {
        for mode in 0..3 {
            let mut w=fixture();let obs=w.observation.clone().unwrap();
            let edges=&mut w.retained.as_mut().unwrap().search.nodes[0].edges;
            if mode==0 { edges[0].outcomes.push((obs.clone(),1)); }
            if mode==1 { edges[0].outcomes[0].1=99; }
            if mode==2 { edges[0].outcomes[0].1=0; }
            w.advance_real_transition(&LostCities,PlayerId::FIRST,PlayerId::FIRST,&A::DrawDeck,&obs);
            assert!(w.retained.is_none());
        }
    }
    #[test]
    fn real_draw_observations_split_own_and_merge_hidden_outcomes() {
        for actor in [PlayerId::FIRST, PlayerId::SECOND] {
            let owner=PlayerId::FIRST;
            let mut state=ready(); state.current_player=actor;
            let action=LostCities.legal_actions(&state).next().unwrap();
            LostCities.apply_action(&mut state,&action).unwrap();
            let before=LostCities.observation(&state,owner);
            let mut nodes=vec![Node {visits:0,edges:vec![]}];
            available(&mut nodes[0], &[A::DrawDeck]);
            let mut observations=Vec::new();
            for seed in 0..20 {
                let mut real=state.clone();
                LostCities.apply_action(&mut real,&A::DrawDeck).unwrap();
                let event=LostCities.sample_chance(&real,&mut SplitMix64::new(seed)).unwrap();
                LostCities.apply_chance_outcome(&mut real,&event).unwrap();
                let obs=LostCities.observation(&real,owner);
                let (child,_) = outcome(&mut nodes,0,0,obs.clone());
                nodes[child].visits=child as u64;
                observations.push((obs,child));
            }
            if actor==owner { assert!(nodes[0].edges[0].outcomes.len()>1); }
            else { assert_eq!(nodes[0].edges[0].outcomes.len(),1); }
            for (obs,child) in observations {
                let mut w=wrapper(BanditPolicy::Uct);w.start_match(&LostCities,owner);
                w.observation=Some(before.clone());
                w.retained=Some(ReuseSearchResult{new_nodes:0,reuse:TreeReuseStats::default(),
                    search:SearchResult{action:A::DrawDeck,diagnostics:Diagnostics::default(),nodes:nodes.clone()}});
                w.advance_real_transition(&LostCities,owner,actor,&A::DrawDeck,&obs);
                assert_eq!(w.pending.transition_hits,1);
                assert_eq!(w.retained.as_ref().unwrap().search.nodes[0].visits,child as u64);
            }
        }
    }
    #[test]
    fn timed_budget_deducts_maintenance_and_reserve() {
        let mut w=wrapper(BanditPolicy::Uct);w.agent.config.budget=SearchBudget::Time(Duration::from_millis(50));
        w.timing.pending=Duration::from_millis(20);w.timing.reserve=Duration::from_millis(10);
        w.begin_decision();assert_eq!(w.timing.allowance,Duration::from_millis(20));
        w.timing.pending=Duration::from_secs(2);w.begin_decision();
        let state=ready();let obs=LostCities.observation(&state,PlayerId::FIRST);
        let r=w.search(&LostCities,&obs,PlayerId::FIRST,&LostCities.legal_actions(&state).collect::<Vec<_>>(),&mut SplitMix64::new(1)).unwrap();
        assert_eq!(r.search.diagnostics.completed_iterations,1);
    }
}
