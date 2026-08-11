use std::{error::Error, fmt};

/// Rejected transition. Creating the message is acceptable because this is an error path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IllegalAction {
    message: String,
}

impl IllegalAction {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for IllegalAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for IllegalAction {}

/// Failure produced while an agent is selecting an action.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentError {
    NoLegalActions,
    Message(String),
}

impl AgentError {
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl fmt::Display for AgentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLegalActions => formatter.write_str("no legal actions are available"),
            Self::Message(message) => formatter.write_str(message),
        }
    }
}

impl Error for AgentError {}
