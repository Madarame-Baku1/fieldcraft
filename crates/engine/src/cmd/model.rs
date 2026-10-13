use serde_json::{Value, json};

use crate::{CommandSpec, Result, Session};

pub const INFO: CommandSpec = CommandSpec { id: "model.info", label: "Model Info", params: "{}", run: info };

fn info(s: &mut Session, _: &Value) -> Result<Value> {
    let counts: serde_json::Map<String, Value> = s.model.element_counts().into_iter().map(|(k, n)| (k.name().to_string(), json!(n))).collect();
    Ok(json!({"nodes": s.model.nodes().len(), "elements": counts}))
}
