//! The command engine.
//!
//! Everything a user can do is a command ([`CommandSpec`]) with a JSON form. The desktop UI,
//! scripts, the CLI and the control channel all run the same commands through
//! [`Session::execute`], which never lets a failure or a panic damage the model: the model is
//! restored and an [`EngineError`] comes back.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]

pub mod cmd;

use modelcraft_model::Model;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EngineError {
    #[error("unknown command `{0}`")]
    UnknownCommand(String),
    #[error("{0}")]
    Params(String),
    #[error(transparent)]
    Model(#[from] modelcraft_model::Error),
    /// A command panicked; the model is unchanged.
    #[error("internal error in {command}: {message}")]
    Internal { command: String, message: String },
}

pub type Result<T> = std::result::Result<T, EngineError>;

/// A command: one user-visible action.
pub struct CommandSpec {
    /// Our own dotted, lower-case id: `<area>.<thing>[.<variant>]`, such as `mesh.node.add`.
    pub id: &'static str,
    pub label: &'static str,
    /// The JSON parameters, documented for scripts, `modelcraft-cli commands` and agents.
    pub params: &'static str,
    pub run: fn(&mut Session, &Value) -> Result<Value>,
}

/// All commands.
pub fn commands() -> &'static [CommandSpec] {
    cmd::ALL
}

pub fn find_command(id: &str) -> Option<&'static CommandSpec> {
    commands().iter().find(|c| c.id == id)
}

/// The model being edited and the commands that built it.
#[derive(Debug, Clone, Default)]
pub struct Session {
    pub model: Model,
    /// Commands that changed the model, in order (a script that rebuilds it).
    pub journal: Vec<(String, Value)>,
}

impl Session {
    /// Run a command. On an error or a panic the model is left as it was.
    pub fn execute(&mut self, id: &str, params: &Value) -> Result<Value> {
        let spec = find_command(id).ok_or_else(|| EngineError::UnknownCommand(id.to_string()))?;
        let before = self.model.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (spec.run)(self, params)));
        let result = match outcome {
            Ok(r) => r,
            Err(p) => {
                let message = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_default();
                log::error!("command {id} panicked: {message}");
                Err(EngineError::Internal { command: id.to_string(), message })
            }
        };
        match &result {
            Ok(_) if self.model != before => self.journal.push((id.to_string(), params.clone())),
            Ok(_) => {}
            Err(_) => self.model = before,
        }
        result
    }

    /// Run a script: a JSON array of `{"command": id, "params": {...}}`.
    pub fn run_script(&mut self, script: &Value) -> Result<()> {
        let steps = script.as_array().ok_or_else(|| EngineError::Params("a script is a JSON array of steps".into()))?;
        for (i, step) in steps.iter().enumerate() {
            let id = step.get("command").and_then(Value::as_str).ok_or_else(|| EngineError::Params(format!("step {}: no `command`", i + 1)))?;
            let params = step.get("params").cloned().unwrap_or(Value::Object(Default::default()));
            self.execute(id, &params).map_err(|e| EngineError::Params(format!("step {} ({id}): {e}", i + 1)))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn command_ids_are_unique_and_well_formed() {
        let mut ids: Vec<&str> = commands().iter().map(|c| c.id).collect();
        for id in &ids {
            assert!(id.split('.').count() >= 2 && id.chars().all(|c| c.is_ascii_lowercase() || c == '.' || c == '_'), "{id}");
        }
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n);
    }

    #[test]
    fn script_builds_a_hex_and_failures_leave_the_model_alone() {
        let mut s = Session::default();
        let mut script = Vec::new();
        for (i, c) in [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]].iter().enumerate() {
            script.push(json!({"command": "mesh.node.add", "params": {"label": i + 1, "xyz": c}}));
        }
        script.push(json!({"command": "mesh.element.add", "params": {"label": 1, "kind": "hex8", "nodes": [1,2,3,4,5,6,7,8]}}));
        s.run_script(&Value::Array(script)).unwrap();
        assert_eq!(s.journal.len(), 9);
        let info = s.execute("model.info", &json!({})).unwrap();
        assert_eq!(info["nodes"], 8);
        assert_eq!(info["elements"]["hex8"], 1);

        let model = s.model.clone();
        assert!(s.execute("mesh.element.add", &json!({"label": 2, "kind": "tet4", "nodes": [1, 2, 3, 99]})).is_err());
        assert!(s.execute("mesh.node.remove", &json!({"label": 1})).is_err());
        assert_eq!(s.model, model);
        assert_eq!(s.journal.len(), 9);
    }

    #[test]
    fn hostile_params_never_panic() {
        let hostile = [
            json!(null),
            json!([]),
            json!("x"),
            json!({}),
            json!({"label": -1}),
            json!({"label": 1e300}),
            json!({"label": u64::MAX}),
            json!({"label": 1, "xyz": [1, 2]}),
            json!({"label": 1, "xyz": ["a", null, {}]}),
            json!({"label": 1, "xyz": [1e308, -1e308, 0]}),
            json!({"label": 1, "kind": "hex8", "nodes": null}),
            json!({"label": 1, "kind": "nope", "nodes": [1]}),
            json!({"label": 1, "kind": "tet4", "nodes": [1, 1, 1, 1]}),
            json!({"label": 1, "kind": "hex20", "nodes": vec![u32::MAX; 20]}),
        ];
        for spec in commands() {
            for p in &hostile {
                let mut s = Session::default();
                let _ = s.execute(spec.id, &p.clone());
            }
        }
        assert!(Session::default().execute("no.such", &json!({})).is_err());
    }
}
