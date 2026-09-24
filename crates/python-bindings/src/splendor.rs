//! Splendor transport. Rule validation and replay remain authoritative Rust operations.
use meeple_bots_catalog::splendor as catalog;
use meeple_bots_core::{Game, PositionStatus};
use meeple_bots_splendor::{Move, Splendor, SplendorAction, SplendorState};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};
fn error(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}
pub fn action_dict(py: Python<'_>, a: SplendorAction) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("type", "splendor")?;
    match a {
        SplendorAction::Refill { card } => {
            d.set_item("kind", "refill")?;
            d.set_item("card", card)?;
        }
        SplendorAction::Play {
            decision,
            returned,
            payment,
            noble,
        } => {
            d.set_item("returned", returned.map(u32::from))?;
            d.set_item("payment", payment.map(u32::from))?;
            d.set_item("noble", noble)?;
            match decision {
                Move::Pass => d.set_item("kind", "pass")?,
                Move::TakeDifferent { colors } => {
                    d.set_item("kind", "take_different")?;
                    d.set_item("colors", colors)?;
                }
                Move::TakeSame { color } => {
                    d.set_item("kind", "take_same")?;
                    d.set_item("color", color)?;
                }
                Move::ReserveVisible { tier, slot } | Move::BuyVisible { tier, slot } => {
                    d.set_item(
                        "kind",
                        if matches!(decision, Move::ReserveVisible { .. }) {
                            "reserve_visible"
                        } else {
                            "buy_visible"
                        },
                    )?;
                    d.set_item("tier", tier)?;
                    d.set_item("slot", slot)?;
                }
                Move::BuyReserved { index } => {
                    d.set_item("kind", "buy_reserved")?;
                    d.set_item("index", index)?;
                }
            }
        }
    }
    Ok(d.unbind())
}
fn parse(d: &Bound<'_, PyDict>) -> PyResult<SplendorAction> {
    fn get<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<T> {
        d.get_item(k)?
            .ok_or_else(|| error(format!("missing {k}")))?
            .extract()
            .map_err(Into::into)
    }
    let kind: String = get(d, "kind")?;
    let decision = match kind.as_str() {
        "pass" => Move::Pass,
        "refill" => {
            return Ok(SplendorAction::Refill {
                card: get(d, "card")?,
            });
        }
        "take_different" => Move::TakeDifferent {
            colors: get(d, "colors")?,
        },
        "take_same" => Move::TakeSame {
            color: get(d, "color")?,
        },
        "reserve_visible" => Move::ReserveVisible {
            tier: get(d, "tier")?,
            slot: get(d, "slot")?,
        },
        "buy_visible" => Move::BuyVisible {
            tier: get(d, "tier")?,
            slot: get(d, "slot")?,
        },
        "buy_reserved" => Move::BuyReserved {
            index: get(d, "index")?,
        },
        _ => return Err(error("unknown Splendor action")),
    };
    Ok(SplendorAction::Play {
        decision,
        returned: get(d, "returned")?,
        payment: get(d, "payment")?,
        noble: get(d, "noble")?,
    })
}
pub fn snapshot(py: Python<'_>, g: &Splendor, s: &SplendorState) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("bank", s.bank.map(u32::from))?;
    d.set_item("market", s.market)?;
    d.set_item(
        "remaining",
        s.remaining
            .iter()
            .map(|pool| pool.iter().copied().map(u32::from).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "nobles",
        s.nobles.iter().copied().map(u32::from).collect::<Vec<_>>(),
    )?;
    d.set_item("active_player", s.active.index())?;
    d.set_item("pending_refill", s.pending_refill)?;
    d.set_item("finished", s.finished)?;
    d.set_item("final_round", s.final_round)?;
    d.set_item("consecutive_passes", s.consecutive_passes)?;
    d.set_item(
        "status",
        match g.status(s) {
            PositionStatus::Chance => "chance",
            PositionStatus::Terminal => "terminal",
            _ => "player_turn",
        },
    )?;
    let players = PyList::empty(py);
    for p in &s.players {
        let v = PyDict::new(py);
        v.set_item("tokens", p.tokens.map(u32::from))?;
        v.set_item("bonuses", p.bonuses.map(u32::from))?;
        v.set_item(
            "purchased",
            p.purchased
                .iter()
                .copied()
                .map(u32::from)
                .collect::<Vec<_>>(),
        )?;
        v.set_item(
            "reserved",
            p.reserved
                .iter()
                .copied()
                .map(u32::from)
                .collect::<Vec<_>>(),
        )?;
        v.set_item(
            "nobles",
            p.nobles.iter().copied().map(u32::from).collect::<Vec<_>>(),
        )?;
        v.set_item("prestige", p.prestige)?;
        players.append(v)?;
    }
    d.set_item("players", players)?;
    Ok(d.unbind())
}
#[pyclass(name = "SplendorPosition", skip_from_py_object)]
#[derive(Clone)]
pub struct PySplendorPosition {
    game: Splendor,
    state: SplendorState,
}
#[pymethods]
impl PySplendorPosition {
    #[new]
    fn new(seed: u64) -> Self {
        let game = catalog::game(seed);
        let state = game.initial_state();
        Self { game, state }
    }
    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        snapshot(py, &self.game, &self.state)
    }
    fn utilities(&self) -> Vec<Option<f32>> {
        [
            meeple_bots_core::PlayerId::FIRST,
            meeple_bots_core::PlayerId::SECOND,
        ]
        .into_iter()
        .map(|p| self.game.terminal_utility(&self.state, p))
        .collect()
    }
    fn legal_actions(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.game
            .legal_actions(&self.state)
            .map(|a| action_dict(py, a))
            .collect()
    }
    fn chance_outcomes(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyDict>, f64)>> {
        self.game
            .chance_outcomes(&self.state)
            .map_err(error)?
            .into_iter()
            .map(|(a, p)| Ok((action_dict(py, a)?, p)))
            .collect()
    }
    fn apply(&self, action: &Bound<'_, PyDict>) -> PyResult<Self> {
        let a = parse(action)?;
        let mut next = self.clone();
        if matches!(a, SplendorAction::Refill { .. }) {
            next.game.apply_chance_outcome(&mut next.state, &a)
        } else {
            next.game.apply_action(&mut next.state, &a)
        }
        .map_err(error)?;
        Ok(next)
    }
}

/// Interactive session with the same seeded RNG streams and lifecycle as catalog matches.
#[pyclass(name = "SplendorSession")]
pub struct PySplendorSession {
    inner: catalog::SplendorSession,
}
#[pymethods]
impl PySplendorSession {
    #[new]
    #[pyo3(signature=(seed=0, first=None, second=None))]
    fn new(
        seed: u64,
        first: Option<&super::PyAgentConfig>,
        second: Option<&super::PyAgentConfig>,
    ) -> PyResult<Self> {
        let automated = |c: Option<&super::PyAgentConfig>| match c.map(|c| &c.inner) {
            None => Ok(None),
            Some(super::PythonAgentConfig::Automated(a)) => Ok(Some(a.as_ref().clone())),
            _ => Err(error("use None for a human session seat")),
        };
        Ok(Self {
            inner: catalog::SplendorSession::new(seed, automated(first)?, automated(second)?)
                .map_err(error)?,
        })
    }
    #[pyo3(signature=(action=None))]
    fn step(&mut self, py: Python<'_>, action: Option<usize>) -> PyResult<Py<PyDict>> {
        py.detach(|| self.inner.step(action)).map_err(error)?;
        self.snapshot(py)
    }
    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let s = &self.inner;
        let d = snapshot(py, &s.game, &s.state)?.into_bound(py);
        // Keep player configuration separate from the authoritative player holdings.
        d.set_item("holdings", d.get_item("players")?.unwrap())?;
        d.del_item("players")?;
        d.set_item("game", "splendor")?;
        d.set_item("phase", d.get_item("status")?.unwrap())?;
        d.set_item("waiting_human", s.waiting_human())?;
        d.set_item(
            "utilities",
            [
                meeple_bots_core::PlayerId::FIRST,
                meeple_bots_core::PlayerId::SECOND,
            ]
            .map(|p| s.game.terminal_utility(&s.state, p)),
        )?;
        d.set_item(
            "legal_actions",
            s.game
                .legal_actions(&s.state)
                .map(|a| action_dict(py, a))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        let events = PyList::empty(py);
        for event in &s.events {
            let e = PyDict::new(py);
            e.set_item("player", event.player.map(|p| p.index()))?;
            e.set_item("action", action_dict(py, event.action)?)?;
            e.set_item("decision_seconds", event.seconds.as_secs_f64())?;
            e.set_item("search_iterations", event.stats.search_iterations)?;
            events.append(e)?;
        }
        d.set_item("events", events)?;
        let cards = PyList::empty(py);
        for c in meeple_bots_splendor::data::CARDS {
            let v = PyDict::new(py);
            v.set_item("id", c.id)?;
            v.set_item("tier", c.tier)?;
            v.set_item("bonus", c.bonus)?;
            v.set_item("points", c.points)?;
            v.set_item("cost", c.cost.map(u32::from))?;
            cards.append(v)?;
        }
        d.set_item("cards", cards)?;
        let nobles = PyList::empty(py);
        for n in meeple_bots_splendor::data::NOBLES {
            let v = PyDict::new(py);
            v.set_item("id", n.id)?;
            v.set_item("points", n.points)?;
            v.set_item("requirements", n.requirements.map(u32::from))?;
            nobles.append(v)?;
        }
        d.set_item("noble_data", nobles)?;
        Ok(d.unbind())
    }
}
