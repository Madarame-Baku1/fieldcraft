use modelcraft_model::{ElementKind, MAX_LABEL};
use serde_json::{Value, json};

use super::{label, xyz};
use crate::{CommandSpec, EngineError, Result, Session};

pub const NODE_ADD: CommandSpec = CommandSpec {
    id: "mesh.node.add",
    label: "Add Node",
    params: r#"{"label": 1, "xyz": [x, y, z]} — adds the node, or moves it if the label exists"#,
    run: node_add,
};

pub const NODE_REMOVE: CommandSpec =
    CommandSpec { id: "mesh.node.remove", label: "Remove Node", params: r#"{"label": 1} — only a node no element uses"#, run: node_remove };

pub const ELEMENT_ADD: CommandSpec = CommandSpec {
    id: "mesh.element.add",
    label: "Add Element",
    params: r#"{"label": 1, "kind": "rod2|beam2|tri3|tri6|quad4|quad8|tet4|tet10|wedge6|hex8|hex20", "nodes": [labels]} — adds or replaces"#,
    run: element_add,
};

pub const ELEMENT_REMOVE: CommandSpec =
    CommandSpec { id: "mesh.element.remove", label: "Remove Element", params: r#"{"label": 1}"#, run: element_remove };

fn node_add(s: &mut Session, p: &Value) -> Result<Value> {
    let l = label(p, "label")?;
    s.model.set_node(l, xyz(p, "xyz")?)?;
    Ok(json!({"label": l}))
}

fn node_remove(s: &mut Session, p: &Value) -> Result<Value> {
    s.model.remove_node(label(p, "label")?)?;
    Ok(Value::Null)
}

fn element_add(s: &mut Session, p: &Value) -> Result<Value> {
    let l = label(p, "label")?;
    let name = p.get("kind").and_then(Value::as_str).unwrap_or_default();
    let kind = ElementKind::from_name(name).ok_or_else(|| EngineError::Params(format!("unknown element kind `{name}`")))?;
    let raw = p.get("nodes").and_then(Value::as_array).ok_or_else(|| EngineError::Params("`nodes` must be a list of node labels".into()))?;
    if raw.len() != kind.node_count() {
        return Err(EngineError::Params(format!("{name} needs {} nodes, got {}", kind.node_count(), raw.len())));
    }
    let nodes = raw
        .iter()
        .map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()).filter(|n| (1..=MAX_LABEL).contains(n)))
        .collect::<Option<Vec<u32>>>()
        .ok_or_else(|| EngineError::Params(format!("`nodes` must be node labels in 1..={MAX_LABEL}")))?;
    s.model.set_element(l, kind, nodes)?;
    Ok(json!({"label": l}))
}

fn element_remove(s: &mut Session, p: &Value) -> Result<Value> {
    s.model.remove_element(label(p, "label")?)?;
    Ok(Value::Null)
}
