"""Reachable fixtures built exclusively through authoritative legal transitions.

These are deliberately simple diagnostic positions, not optimal-move labels.
Late cases retain focal cards but acquire additional public history and other cards.
"""
from functools import partial
from random import Random

from ...lost_cities import LostCities, LostCitiesAction as Action
from ..core import ProbeCase, ProbePosition

GAME = LostCities()
COLORS = ('Red', 'Green', 'Blue', 'Yellow', 'White')
SEEDS = {'ordering': 1647, 'jump': 3404, 'marginal': 22, 'support': 338,
         'unsupported': 12, 'discard': 187, 'draw': 34}


def label(action):
    if isinstance(action, dict):
        action = Action.from_dict(action)
    if action.card is not None:
        color, value = action.card
        return f'{action.kind.title()} {COLORS[color]} {value if value else "Wager"}'
    return 'Draw deck' if action.kind == 'draw_deck' else f'Draw {COLORS[action.color]} discard'


def build_state(name):
    """Return authoritative state plus replayable fixture provenance."""
    family = ('ordering' if name.startswith('preserve_') else 'jump' if name.startswith('avoid_') else
              'marginal' if name.startswith('marginal_') else 'unsupported' if name == 'wager_without_support' else
              'support' if name == 'wager_with_support' else 'draw' if name == 'discard_pile_vs_deck' else 'discard')
    seed = SEEDS[family]
    state = GAME.initial_state(seed)
    rng = Random(seed)
    transitions = []

    def apply(action, chance=False):
        nonlocal state
        transitions.append({'chance': chance, 'action': action.to_dict()})
        state = (GAME.apply_chance_outcome if chance else GAME.apply_action)(state, action)

    def turn(action):
        actor = state.current_player
        apply(action)
        apply(Action('draw_deck'))
        outcomes = [a for a, _ in GAME.chance_outcomes(state)]
        # Fixed environment choice, independent of search RNG. Keep red draws out
        # of the focal hand so marginal/ordering contrasts retain their structure.
        preferred = [a for a in outcomes if (a.card[0] != 0) == (actor == 0)]
        apply(rng.choice(preferred or outcomes), chance=True)

    def discard():
        return next(a for a in GAME.legal_actions(state) if a.kind == 'discard'
                    and (state.current_player != 0 or a.card[0] != 0))

    if family == 'jump':
        turn(Action('play', (0, 3)))
        turn(discard())
    elif family == 'discard':
        turn(discard())
        turn(Action('play', (0, 3 if name == 'dangerous_discard' else 10)))
    elif family == 'draw':
        turn(discard())
        turn(Action('discard', (0, 6)))
        apply(discard())
    if name.endswith('_late'):
        while len(state.deck) > 4:
            turn(discard())
    assert state.current_player == 0 and state.status == 'player'
    return state, {'initial_seed': seed, 'transitions': transitions, 'state': state.to_dict()}


def position(name):
    state, fixture = build_state(name)
    return ProbePosition(GAME.observation(state, 0), GAME.legal_actions(state), 0, fixture)


def case(name, description, tags, candidates, notes):
    return ProbeCase('lost_cities.' + name, 'lost_cities', description, tuple(tags),
                     partial(position, name), label, tuple(candidates), notes)


def plays(*values):
    return [Action('play', (0, v)) for v in values]


CASES = (
    case('preserve_low_sequence_early', 'Ascending red cards with a full draw deck.',
         ['ordering', 'early-game', 'optionality'], plays(4, 6, 8),
         'Playing a lower card preserves future optionality; this is a diagnostic hypothesis.'),
    case('preserve_low_sequence_late', 'Ascending red cards with four deck cards remaining.',
         ['ordering', 'late-game', 'optionality'], plays(4, 6, 8),
         'Contrast with early ordering; public discards and non-focal hand cards also differ.'),
    case('avoid_large_irreversible_jump', 'Red expedition at 3; hand contains 5 and 9.',
         ['ordering', 'optionality'], plays(5, 9), 'Measure sensitivity to irreversible jumps.'),
    case('marginal_expedition_early', 'Only Red 4 supports a new red expedition.',
         ['commitment', 'early-game'], [*plays(4), Action('discard', (0, 4))], 'Starting an expedition incurs its normal cost.'),
    case('marginal_expedition_late', 'Only Red 4 supports red; four deck cards remain.',
         ['commitment', 'late-game'], [*plays(4), Action('discard', (0, 4))], 'Contrast commitment under limited remaining turns.'),
    case('wager_with_support', 'Red wager with Red 4, 6 and 9 in hand.',
         ['wagers'], [*plays(0, 4), Action('discard', (0, 0))], 'Measure wager sensitivity to supporting cards.'),
    case('wager_without_support', 'Red wager without red numbered cards in hand.',
         ['wagers'], [*plays(0), Action('discard', (0, 0))], 'No optimal action is prescribed.'),
    case('dangerous_discard', 'Opponent red expedition is at 3; own hand contains Red 9.',
         ['discard'], [*plays(9), Action('discard', (0, 9))], 'Red 9 could advance the public opposing expedition.'),
    case('safer_discard', 'Opponent red expedition is at 10; own hand contains Red 9.',
         ['discard'], [*plays(9), Action('discard', (0, 9))], 'Opponent cannot play Red 9 above its public Red 10.'),
    case('discard_pile_vs_deck', 'Draw phase: visible Red 6 versus hidden deck; own Red 4.',
         ['draw'], [Action('draw_discard', color=0), Action('draw_deck')],
         'Compare taking a visible compatible card with an unknown draw.'),
)
