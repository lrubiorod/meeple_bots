//! Explicit authoritative position and value-only player observation transports.
use meeple_bots_core::{Game, ImperfectInformationGame, PlayerId, PositionStatus};
use meeple_bots_lost_cities::{
    LostCities, LostCitiesAction as Action, LostCitiesCard as Card, LostCitiesColor as Color,
    LostCitiesObservation as Observation, LostCitiesSimulationWorld, LostCitiesState as State,
    Phase,
};
use meeple_bots_simulation::SplitMix64;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
fn error(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn get<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<T> {
    d.get_item(k)?
        .ok_or_else(|| error(format!("missing {k}")))?
        .extract()
        .map_err(Into::into)
}
fn player(n: usize) -> PyResult<PlayerId> {
    match n {
        0 => Ok(PlayerId::FIRST),
        1 => Ok(PlayerId::SECOND),
        _ => Err(error("player must be 0 or 1")),
    }
}
fn color(n: usize) -> PyResult<Color> {
    Color::ALL
        .get(n)
        .copied()
        .ok_or_else(|| error("invalid color"))
}
fn card(raw: [u8; 2]) -> PyResult<Card> {
    let c = color(usize::from(raw[0]))?;
    let card = if raw[1] == 0 {
        Card::Wager(c)
    } else {
        Card::Number(c, raw[1])
    };
    if card.valid() {
        Ok(card)
    } else {
        Err(error("invalid card value"))
    }
}
fn cards(raw: Vec<[u8; 2]>) -> PyResult<Vec<Card>> {
    raw.into_iter().map(card).collect()
}
fn raw(cards: &[Card]) -> Vec<[u8; 2]> {
    cards.iter().map(|c| [c.color() as u8, c.value()]).collect()
}
fn phase_name(p: Phase) -> &'static str {
    match p {
        Phase::Deal(_) => "deal",
        Phase::Play => "play",
        Phase::Draw => "draw",
        Phase::DrawChance => "draw_chance",
        Phase::Finished => "finished",
    }
}
pub fn action_dict(py: Python<'_>, a: Action) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("type", "lost_cities")?;
    let (kind, c) = match a {
        Action::Play(c) => ("play", Some(c)),
        Action::Discard(c) => ("discard", Some(c)),
        Action::DealCard(c) => ("deal_card", Some(c)),
        Action::DrawDeck => ("draw_deck", None),
        Action::DrawDiscard(c) => {
            d.set_item("color", c as usize)?;
            ("draw_discard", None)
        }
    };
    d.set_item("kind", kind)?;
    if let Some(c) = c {
        d.set_item("card", [c.color() as u8, c.value()])?;
    }
    Ok(d.unbind())
}
fn parse_action(d: &Bound<'_, PyDict>) -> PyResult<Action> {
    if get::<String>(d, "type")? != "lost_cities" {
        return Err(error("expected Lost Cities action"));
    }
    match get::<String>(d, "kind")?.as_str() {
        "play" => Ok(Action::Play(card(get(d, "card")?)?)),
        "discard" => Ok(Action::Discard(card(get(d, "card")?)?)),
        "deal_card" => Ok(Action::DealCard(card(get(d, "card")?)?)),
        "draw_deck" => Ok(Action::DrawDeck),
        "draw_discard" => Ok(Action::DrawDiscard(color(get(d, "color")?)?)),
        _ => Err(error("invalid action kind")),
    }
}
fn observation_dict(py: Python<'_>, o: &Observation) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("observer", o.observer.index())?;
    d.set_item("current_player", o.current_player.index())?;
    d.set_item("phase", phase_name(o.phase))?;
    d.set_item(
        "dealt",
        if let Phase::Deal(n) = o.phase {
            Some(n)
        } else {
            None
        },
    )?;
    d.set_item("hand", raw(&o.hand))?;
    d.set_item("opponent_hand_size", o.opponent_hand_size)?;
    d.set_item("deck_size", o.deck_size)?;
    d.set_item(
        "expeditions",
        o.expeditions
            .iter()
            .map(|e| e.iter().map(|p| raw(p)).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "discards",
        o.discards.iter().map(|p| raw(p)).collect::<Vec<_>>(),
    )?;
    d.set_item("blocked_discard", o.blocked_discard.map(|c| c as usize))?;
    Ok(d.unbind())
}
fn parse_observation(d: &Bound<'_, PyDict>) -> PyResult<Observation> {
    let phase = match get::<String>(d, "phase")?.as_str() {
        "deal" => Phase::Deal(get(d, "dealt")?),
        "play" => Phase::Play,
        "draw" => Phase::Draw,
        "draw_chance" => Phase::DrawChance,
        "finished" => Phase::Finished,
        _ => return Err(error("invalid phase")),
    };
    let expeditions: Vec<Vec<Vec<[u8; 2]>>> = get(d, "expeditions")?;
    let expeditions = expeditions
        .into_iter()
        .map(|p| {
            p.into_iter()
                .map(cards)
                .collect::<PyResult<Vec<_>>>()?
                .try_into()
                .map_err(|_| error("expected five expeditions"))
        })
        .collect::<PyResult<Vec<[Vec<Card>; 5]>>>()?
        .try_into()
        .map_err(|_| error("expected two players"))?;
    let discards = get::<Vec<Vec<[u8; 2]>>>(d, "discards")?
        .into_iter()
        .map(cards)
        .collect::<PyResult<Vec<_>>>()?
        .try_into()
        .map_err(|_| error("expected five discards"))?;
    Ok(Observation {
        observer: player(get(d, "observer")?)?,
        current_player: player(get(d, "current_player")?)?,
        phase,
        hand: cards(get(d, "hand")?)?,
        opponent_hand_size: get(d, "opponent_hand_size")?,
        deck_size: get(d, "deck_size")?,
        expeditions,
        discards,
        blocked_discard: get::<Option<usize>>(d, "blocked_discard")?
            .map(color)
            .transpose()?,
    })
}
pub fn snapshot(py: Python<'_>, s: &State) -> PyResult<Py<PyDict>> {
    let d = observation_dict(py, &LostCities.observation(s, PlayerId::FIRST))?.into_bound(py);
    for key in ["observer", "hand", "opponent_hand_size", "deck_size"] {
        d.del_item(key)?;
    }
    d.set_item("hands", s.hands.iter().map(|h| raw(h)).collect::<Vec<_>>())?;
    d.set_item("deck", raw(&s.deck))?;
    d.set_item("scores", s.scores())?;
    d.set_item(
        "status",
        match LostCities.status(s) {
            PositionStatus::Chance => "chance",
            PositionStatus::Terminal => "terminal",
            _ => "player",
        },
    )?;
    Ok(d.unbind())
}
#[pyclass(name = "LostCitiesPosition", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyLostCitiesPosition {
    state: State,
}
#[pymethods]
impl PyLostCitiesPosition {
    #[staticmethod]
    fn replay(py: Python<'_>, moves: Vec<Py<PyDict>>, events: Vec<Py<PyDict>>) -> PyResult<Self> {
        let mut state = LostCities.initial_state();
        let mut pending = events.iter().peekable();
        for i in 0..=moves.len() {
            while let Some(event) = pending.peek() {
                if get::<usize>(event.bind(py), "after_ply")? != i {
                    break;
                }
                let outcome: Bound<'_, PyDict> = get(event.bind(py), "outcome")?;
                LostCities
                    .apply_chance_outcome(&mut state, &parse_action(&outcome)?)
                    .map_err(error)?;
                pending.next();
            }
            if let Some(m) = moves.get(i) {
                let p = player(get(m.bind(py), "player")?)?;
                if LostCities.status(&state) != PositionStatus::PlayerTurn(p) {
                    return Err(error("incorrect player or transition order"));
                }
                let action: Bound<'_, PyDict> = get(m.bind(py), "action")?;
                LostCities
                    .apply_action(&mut state, &parse_action(&action)?)
                    .map_err(error)?;
            }
        }
        if pending.next().is_some() || LostCities.status(&state) != PositionStatus::Terminal {
            return Err(error("missing or extra transitions"));
        }
        LostCities.validate_state(&state).map_err(error)?;
        Ok(Self { state })
    }
    #[new]
    #[pyo3(signature=(seed=0))]
    fn new(seed: u64) -> PyResult<Self> {
        let mut state = LostCities.initial_state();
        let mut rng = SplitMix64::new(seed ^ 0x8EBC_6AF0_9C88_C6E3);
        while LostCities.status(&state) == PositionStatus::Chance {
            let a = LostCities.sample_chance(&state, &mut rng).map_err(error)?;
            LostCities
                .apply_chance_outcome(&mut state, &a)
                .map_err(error)?;
        }
        Ok(Self { state })
    }
    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        snapshot(py, &self.state)
    }
    fn observation(&self, py: Python<'_>, observer: usize) -> PyResult<Py<PyDict>> {
        observation_dict(py, &LostCities.observation(&self.state, player(observer)?))
    }
    #[staticmethod]
    fn determinize(
        observation: &Bound<'_, PyDict>,
        observer: usize,
        seed: u64,
    ) -> PyResult<PyLostCitiesWorld> {
        let o = parse_observation(observation)?;
        Ok(PyLostCitiesWorld {
            world: LostCities
                .sample_determinization(&o, player(observer)?, &mut SplitMix64::new(seed))
                .map_err(error)?,
        })
    }
    fn legal_actions(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        LostCities
            .legal_actions(&self.state)
            .map(|a| action_dict(py, a))
            .collect()
    }
    fn chance_outcomes(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyDict>, f64)>> {
        LostCities
            .chance_outcomes(&self.state)
            .map_err(error)?
            .into_iter()
            .map(|(a, p)| Ok((action_dict(py, a)?, p)))
            .collect()
    }
    fn sample_chance(&self, py: Python<'_>, seed: u64) -> PyResult<Py<PyDict>> {
        action_dict(
            py,
            LostCities
                .sample_chance(&self.state, &mut SplitMix64::new(seed))
                .map_err(error)?,
        )
    }
    fn apply_action(&self, action: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut result = self.clone();
        LostCities
            .apply_action(&mut result.state, &parse_action(action)?)
            .map_err(error)?;
        Ok(result)
    }
    fn apply_chance_outcome(&self, outcome: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut result = self.clone();
        LostCities
            .apply_chance_outcome(&mut result.state, &parse_action(outcome)?)
            .map_err(error)?;
        Ok(result)
    }
    fn validate(&self) -> PyResult<()> {
        LostCities.validate_state(&self.state).map_err(error)
    }
}

#[pyclass(name = "LostCitiesSimulationWorld", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyLostCitiesWorld {
    world: LostCitiesSimulationWorld,
}
#[pymethods]
impl PyLostCitiesWorld {
    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        snapshot(py, self.world.state())
    }
    fn deck_order(&self) -> Vec<[u8; 2]> {
        raw(self.world.deck_order())
    }
    fn observation(&self, py: Python<'_>, observer: usize) -> PyResult<Py<PyDict>> {
        observation_dict(py, &self.world.observation(player(observer)?))
    }
    fn legal_actions(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.world
            .legal_actions()
            .map(|a| action_dict(py, a))
            .collect()
    }
    fn apply_action(&self, action: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut result = self.clone();
        result
            .world
            .apply_action(&parse_action(action)?)
            .map_err(error)?;
        Ok(result)
    }
    fn resolve_pending_draws(&self) -> PyResult<Self> {
        let mut result = self.clone();
        result.world.resolve_pending_draws().map_err(error)?;
        Ok(result)
    }
    fn validate(&self) -> PyResult<()> {
        self.world.validate().map_err(error)
    }
}
