use std::fmt;

/// Stable, zero-based identifier for a player within a game.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PlayerId(u8);

impl PlayerId {
    pub const FIRST: Self = Self(0);
    pub const SECOND: Self = Self(1);

    pub const fn new(index: u8) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for PlayerId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}
