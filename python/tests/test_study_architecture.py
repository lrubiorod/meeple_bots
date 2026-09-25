"""Phase 2 ownership boundaries without strategic outcome assertions."""
import copy
import unittest

from meeple_bots import MctsAgent, studies
from meeple_bots._study_profiles import PROFILES as old_profiles
from meeple_bots._study_tuners import proposals as old_proposals
from meeple_bots.studies import budget, coordinator, persistence, planning, profiles, race, report, tuners


class StudyArchitectureTests(unittest.TestCase):
    def test_facade_and_private_compatibility_paths_keep_object_identity(self):
        self.assertIs(studies.StudyRunner, coordinator.StudyRunner)
        self.assertIs(studies.run_study, coordinator.run_study)
        self.assertIs(studies.profile_values, persistence.profile_values)
        self.assertIs(studies._group_leaders, race._group_leaders)
        self.assertIs(studies._build_phase, planning._build_phase)
        self.assertIs(old_profiles, profiles.PROFILES)
        self.assertIs(old_proposals, tuners.proposals)
        self.assertFalse(hasattr(profiles.PROFILES['so_ismcts'], 'calibrate'))
        self.assertFalse(hasattr(profiles.PROFILES['so_ismcts'], 'build_phase'))

    def test_budget_allocation_and_extension_notice_do_not_mutate_inputs(self):
        agent = persistence.profile_values(MctsAgent(iterations=4))
        phase = {'name': 'exploration', 'agents': {'a': agent, 'b': {**agent, 'exploration': 1.0}},
                 'contrasts': [{'a': 'a', 'b': 'b'}]}
        request = {'max_pairs': 4, 'stage_games': {}, 'workers': 1}
        before = copy.deepcopy(phase)
        planned = budget.plan_phase(phase, request, 100, [1.0], [1], planning.budget_stage(phase))
        self.assertEqual(phase, before)
        self.assertEqual(planned['contrasts'][0]['target_pairs'], 4)
        self.assertEqual(planned['planned_games'], 8)
        self.assertEqual(planned['allocation_mode'], 'fixed_games')

        extension = {'name': 'rave_extend_3', 'agents': {'a': {'rave_equivalence': 3},
                     'b': {'rave_equivalence': 6}}, 'groups': {'main': ['a', 'b']},
                     'contrasts': [{'a': 'a', 'b': 'b', 'target_pairs': 4,
                                    'result': {'seed_pairs': 4, 'score_b': .8,
                                               'seed_scores_b': {'0': 1, '1': 1, '2': .5, '3': .7}}}]}
        state = {'phases': {'rave': {'agents': {'old': {'rave_equivalence': 1}}},
                            'rave_extend_3': extension}}
        frozen = copy.deepcopy(state)
        message, limit = report.extension_notice(state, extension, completed=True)
        self.assertTrue(limit)
        self.assertIn('extension limit (3/3)', message)
        self.assertEqual(state, frozen)


if __name__ == '__main__':
    unittest.main()
