# NAFEMS LE2 cylindrical shell bending patch test (MITC4 shells)

Our model of the published problem: a 30° sector of a cylindrical shell of mid-surface radius
1000 mm, axial length 500 mm and thickness 10 mm, E = 210 GPa, ν = 0.3 (N, mm, MPa). The straight
edge AB is clamped, the curved edges AD and BC keep their axial displacement (planes of symmetry
across the axis), and the other straight edge DC carries the load. Two load cases, one step each:

1. **Moment**: a uniform edge moment of 1000 N·mm/mm on DC about the axis;
2. **Pressure**: 0.6 MPa outward on the mid-surface, with a hoop traction of 60 MPa on DC (600 N/mm
   over the thickness), the membrane force that balances it.

Both put the patch in a uniform state with a hoop stress of 60 MPa on the outer surface (6M/t² in
bending, pR/t in membrane), the published target at E. The descriptions we read do not place E;
since the exact stress is the same everywhere on the patch, we read it at the patch centre
(250, 0, 1000), the node on the y = 0 plane where the hoop direction is the global y axis, so the
tangential stress is σ_yy.

## The mesh

The patch is curved, so the recipe writes its 81 nodes and 64 quad4 elements one by one with
`mesh.node.add` and `mesh.element.add` ([docs/verification.md](../../../docs/verification.md)).
The axis is x and the arc runs from φ = −15° (AB) to φ = 15° (DC): node 1 + i + 9 j lies at
(500 i/8, 1000 sin φ_j, 1000 cos φ_j) with φ_j = −15° + 30° j/8, and element 1 + i + 8 j is
[node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)], so its normal points away from
the axis and the shell's top surface is the outer one (the surface nodal stresses report).

The edge moment and edge traction are written as consistent nodal loads on DC: an end node of the
edge takes half of an inner node's share (two loads with `"distribution": "per_node"`). The
pressure acts on the elements' inner faces (face 2, the reversed node order), pushing outward.

The recipe was generated with this script (`python3 le2.py 8 > recipe.json`):

```python
import json, math, sys

def le2(n):
    """NAFEMS LE2 cylindrical patch (mm, N, MPa): R = 1000, axial length 500, 30° arc from φ = −15° (AB) to 15° (DC)."""
    m = n  # elements round the arc, as many as along the axis
    R, L, half = 1000.0, 500.0, math.radians(15.0)
    label = lambda i, j: 1 + i + (n + 1) * j
    commands = []
    for j in range(m + 1):
        for i in range(n + 1):
            phi = -half + 2 * half * j / m
            commands.append({"command": "mesh.node.add", "params": {"label": label(i, j), "xyz": [L * i / n, R * math.sin(phi), R * math.cos(phi)]}})
    for j in range(m):
        for i in range(n):
            # i along the axis, j round the arc: e3 points away from the axis, so top is the outer surface
            nodes = [label(i, j), label(i + 1, j), label(i + 1, j + 1), label(i, j + 1)]
            commands.append({"command": "mesh.element.add", "params": {"label": 1 + i + n * j, "kind": "quad4", "nodes": nodes}})
    h = L / n
    ends = [label(0, m), label(n, m)]
    inner = [label(i, m) for i in range(1, n)]
    t = [0.0, math.cos(half), -math.sin(half)]  # the hoop direction at DC, away from AB
    return commands + [
        {"command": "model.set.create", "params": {"name": "patch", "elements": list(range(1, n * m + 1))}},
        {"command": "model.set.create", "params": {"name": "inside", "faces": [{"element": e, "face": 2} for e in range(1, n * m + 1)]}},
        {"command": "model.set.create", "params": {"name": "AB", "nodes": [label(i, 0) for i in range(n + 1)]}},
        {"command": "model.set.create", "params": {"name": "AD_BC", "nodes": sorted([label(0, j) for j in range(m + 1)] + [label(n, j) for j in range(m + 1)])}},
        {"command": "model.set.create", "params": {"name": "DC_ends", "nodes": ends}},
        {"command": "model.set.create", "params": {"name": "DC_inner", "nodes": inner}},
        {"command": "model.material.create", "params": {"name": "Steel", "elastic": {"type": "isotropic", "young": 210000.0, "poisson": 0.3}}},
        {"command": "model.section.create", "params": {"name": "patch", "set": "patch", "material": "Steel", "kind": {"type": "shell", "thickness": 10.0}}},
        {"command": "model.boundary_condition.create", "params": {"name": "AB", "type": "displacement", "target": {"set": "AB"}, "values": [0, 0, 0, 0, 0, 0]}},
        {"command": "model.boundary_condition.create", "params": {"name": "AD_BC", "type": "symmetry", "target": {"set": "AD_BC"}, "plane": "x"}},
        # load case 1: an edge moment of 1000 N·mm/mm on DC about the axis, as consistent nodal moments
        {"command": "model.load.create", "params": {"name": "moment_ends", "type": "moment", "target": {"set": "DC_ends"}, "vector": [-1000.0 * h / 2, 0, 0], "distribution": "per_node"}},
        {"command": "model.load.create", "params": {"name": "moment_inner", "type": "moment", "target": {"set": "DC_inner"}, "vector": [-1000.0 * h, 0, 0], "distribution": "per_node"}},
        # load case 2: 0.6 MPa outward on the mid-surface, 60 MPa × 10 mm hoop traction on DC
        {"command": "model.load.create", "params": {"name": "pressure", "type": "pressure", "set": "inside", "magnitude": 0.6}},
        {"command": "model.load.create", "params": {"name": "pull_ends", "type": "force", "target": {"set": "DC_ends"}, "vector": [0.0, 600.0 * h / 2 * t[1], 600.0 * h / 2 * t[2]], "distribution": "per_node"}},
        {"command": "model.load.create", "params": {"name": "pull_inner", "type": "force", "target": {"set": "DC_inner"}, "vector": [0.0, 600.0 * h * t[1], 600.0 * h * t[2]], "distribution": "per_node"}},
        {"command": "model.step.create", "params": {"name": "Moment", "procedure": {"type": "static"}, "boundary_conditions": ["AB", "AD_BC"], "loads": ["moment_ends", "moment_inner"]}},
        {"command": "model.step.create", "params": {"name": "Pressure", "procedure": {"type": "static"}, "boundary_conditions": ["AB", "AD_BC"], "loads": ["pressure", "pull_ends", "pull_inner"]}},
        {"command": "solve.run", "params": {}, "id": "solve"},
    ]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
print("[\n " + ",\n ".join(json.dumps(c) for c in le2(n)) + "\n]")
```

## Refinement

Every row is the listing above with the row's divisions, solved and probed at E as the case
does; the case mesh is 8 × 8:

| cells | σ_yy at E, moment | error | σ_yy at E, pressure | error |
|---|---|---|---|---|
| 4 × 4 | 59.743 | −0.43% | 59.615 | −0.64% |
| 8 × 8 (this case) | 59.936 | −0.11% | 59.904 | −0.16% |
| 16 × 16 | 59.984 | −0.03% | 59.976 | −0.04% |
| 32 × 32 | 59.996 | −0.01% | 59.994 | −0.01% |

The error falls as the square of the facet angle: each flat element is tilted by half its arc
angle from the hoop direction at E, so a share sin²(Δφ/2) of the hoop stress shows in σ_zz instead
(σ_yy + σ_zz is 60.000 in every row of load case 1). The next refinement of the case mesh moves the
value by 0.05% (moment) and 0.07% (pressure), well under a tenth of the 1% tolerance.

## Tolerance

1%: the target is exact for a thin shell and FieldCraft converges to it; the 8 × 8 mesh carries
a 0.1–0.2% facet error, a fifth of the tolerance or less. `fixture.json` holds the 8 × 8 values:
the mesh is written by the recipe, so it is bit-identical on every platform.

## Running it

```sh
cargo xtask oracle --case nafems-le2
cargo xtask oracle --bless nafems-le2   # only after the case passes
```
