# NAFEMS LE3 hemispherical shell with point loads (MITC4 shells)

Our model of the published problem: a quarter of a closed hemisphere of mid-surface radius 10 m
and thickness 0.04 m, E = 68.25 GPa, ν = 0.3 (m, kN, kPa: E = 6.825 × 10⁷), centred at the origin
with its pole E on +z. A radial force of 2 kN pulls outward at A (10, 0, 0) on the equator and one
of 2 kN pushes inward at C (0, 10, 0). The edge AE lies on the plane y = 0 and CE on x = 0, both
planes of symmetry; u_z = 0 at E holds the shell vertically; the equator is free. The target is
the radial displacement at A, u_x there.

(LE3 has no hole at the pole, unlike the MacNeal–Harder hemisphere of
`pinched-hemisphere`, which has an 18° hole and different loads.)

## The mesh

The surface is doubly curved, so the recipe writes its nodes and elements one by one with
`mesh.node.add` and `mesh.element.add` ([docs/verification.md](../../../docs/verification.md)).
Meridians every 90°/32 of longitude λ and parallels every 90°/32 of latitude β divide the quarter
into cells; node 1 + i + 33 j lies at R (cos β_j cos λ_i, cos β_j sin λ_i, sin β_j), and the pole is
node 1057. Every cell below the top parallel is a quad4 whose four corners lie on two parallels,
symmetric about its mid-meridian, so it is planar (flat shells model a curved surface best with
flat facets); the 32 cells at the pole are tri3 (MITC3) triangles that share the pole node. All
elements run counter-clockwise seen from outside.

A first mesh projected three faces of a cube onto the sphere (no pole, all quad4), but its cells
are warped, and the flat MITC4 element on warped cells was far too stiff (0.147 at 32 × 32 per
face). The longitude–latitude mesh converges as the table shows.

The recipe was generated with this script (`python3 le3.py 32 > recipe.json`):

```python
import json, math, sys

def le3(n):
    """NAFEMS LE3 hemisphere quadrant (m, kN, kPa): R = 10, x ≥ 0, y ≥ 0, z ≥ 0, pole E on +z.

    Meridians every 90°/n of longitude λ, parallels every 90°/n of latitude β; the cells are planar
    trapezoids and the ring at the pole is triangles."""
    R = 10.0
    label = lambda i, j: 1 + i + (n + 1) * j  # longitude i, latitude j < n
    pole = 1 + (n + 1) * n
    commands = []
    for j in range(n):
        for i in range(n + 1):
            lam, beta = math.pi / 2 * i / n, math.pi / 2 * j / n
            xyz = [R * math.cos(beta) * math.cos(lam), R * math.cos(beta) * math.sin(lam), R * math.sin(beta)]
            commands.append({"command": "mesh.node.add", "params": {"label": label(i, j), "xyz": xyz}})
    commands.append({"command": "mesh.node.add", "params": {"label": pole, "xyz": [0.0, 0.0, R]}})
    k = 0
    for j in range(n):
        for i in range(n):
            k += 1
            if j < n - 1:
                kind, nodes = "quad4", [label(i, j), label(i + 1, j), label(i + 1, j + 1), label(i, j + 1)]
            else:
                kind, nodes = "tri3", [label(i, j), label(i + 1, j), pole]
            commands.append({"command": "mesh.element.add", "params": {"label": k, "kind": kind, "nodes": nodes}})
    A, C = label(0, 0), label(n, 0)
    return commands + [
        {"command": "model.set.create", "params": {"name": "shell", "elements": list(range(1, k + 1))}},
        {"command": "model.set.create", "params": {"name": "AE", "nodes": [label(0, j) for j in range(n)] + [pole]}},
        {"command": "model.set.create", "params": {"name": "CE", "nodes": [label(n, j) for j in range(n)] + [pole]}},
        {"command": "model.set.create", "params": {"name": "A", "nodes": [A]}},
        {"command": "model.set.create", "params": {"name": "C", "nodes": [C]}},
        {"command": "model.set.create", "params": {"name": "E", "nodes": [pole]}},
        {"command": "model.material.create", "params": {"name": "Alloy", "elastic": {"type": "isotropic", "young": 6.825e7, "poisson": 0.3}}},
        {"command": "model.section.create", "params": {"name": "shell", "set": "shell", "material": "Alloy", "kind": {"type": "shell", "thickness": 0.04}}},
        {"command": "model.boundary_condition.create", "params": {"name": "AE", "type": "symmetry", "target": {"set": "AE"}, "plane": "y"}},
        {"command": "model.boundary_condition.create", "params": {"name": "CE", "type": "symmetry", "target": {"set": "CE"}, "plane": "x"}},
        {"command": "model.boundary_condition.create", "params": {"name": "E", "type": "displacement", "target": {"set": "E"}, "values": [None, None, 0, None, None, None]}},
        {"command": "model.load.create", "params": {"name": "pull_A", "type": "force", "target": {"set": "A"}, "vector": [2.0, 0, 0]}},
        {"command": "model.load.create", "params": {"name": "push_C", "type": "force", "target": {"set": "C"}, "vector": [0, -2.0, 0]}},
        {"command": "model.step.create", "params": {"name": "Load", "procedure": {"type": "static"}, "boundary_conditions": ["AE", "CE", "E"], "loads": ["pull_A", "push_C"]}},
        {"command": "solve.static", "params": {}, "id": "solve"},
    ]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
print("[\n " + ",\n ".join(json.dumps(c) for c in le3(n)) + "\n]")
```

## Refinement

Every row is the listing above with the row's divisions, solved and probed at A as the case does;
the case mesh is 32 × 32:

| cells | u_x at A (m) | error |
|---|---|---|
| 8 × 8 | 0.18197 | −1.64% |
| 16 × 16 | 0.18324 | −0.95% |
| 32 × 32 (this case) | 0.18418 | −0.44% |
| 64 × 64 | 0.18469 | −0.17% |

The result rises with every refinement and its increments shrink, but slowly: under a point load,
a shell with transverse shear flexibility has no finite limit, only a very slow (logarithmic)
growth (the ESRD guide makes the same remark for its own p-version solution, 0.1844). The next
refinement moves the case value by 0.28%, a seventh of the tolerance.

## Tolerance

2%: the target is published to three figures and the displacement under a point load is not a
converging quantity in the strict sense (above); 32 × 32 is within 0.44%. `fixture.json` holds the
32 × 32 value: the mesh is written by the recipe, so it is bit-identical on every platform.

## Running it

```sh
cargo xtask oracle --case nafems-le3
cargo xtask oracle --bless nafems-le3   # only after the case passes
```
