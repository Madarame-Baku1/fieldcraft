# NAFEMS LE6 skew plate under normal pressure (MITC4 shells)

Our model of the published problem: a rhombic plate with 1 m sides and a 30° skew angle, 10 mm
thick, E = 210 GPa, ν = 0.3 (N, mm, MPa), under a uniform pressure of 0.7 kPa downward. The corners
are A (0, 0), B (1000, 0), C (1866.03, 500) and D (866.03, 500): acute angles of 30° at A and C,
obtuse angles of 150° at B and D. All four edges are simply supported (u_z = 0, rotations free);
u_x = u_y = 0 at A and u_y = 0 at B hold the plate in its plane. The target is the maximum
principal stress on the lower surface at E, the centre of the plate (933.01, 250).

## The mesh

The recipe writes the 4225 nodes and 4096 quad4 elements one by one with `mesh.node.add` and
`mesh.element.add` ([docs/verification.md](../../../docs/verification.md)); a skewed grid is
simpler to write than to block-mesh. Grid line k of 64 lies at (1 − cos(π k/64))/2 of the side in
each direction, so the cells shrink towards every edge, and most at the obtuse corners B and D,
where the plate's bending moments are singular; a uniform grid converges much more slowly there
(below). E is node 2113, the middle of the grid. The elements run clockwise seen from +z, so each
element's normal points down and the shell's top surface, which the nodal stresses report, is
the plate's lower surface. The pressure is a load per volume over the thickness (0.7 kPa / 10 mm),
the same uniform load on a flat plate.

The recipe was generated with this script (`python3 le6.py 64 > recipe.json`):

```python
import json, math, sys

def le6(n):
    """NAFEMS LE6 skew plate (mm, N, MPa): A (0, 0), B (1000, 0), C = B + s, D = A + s with
    s = 1000 (cos 30°, sin 30°); n × n cells graded towards every edge (cosine spacing), so the
    obtuse corners B and D, where the plate's stresses are singular, get the smallest cells."""
    s = (1000.0 * math.cos(math.pi / 6), 1000.0 * math.sin(math.pi / 6))
    g = lambda k: (1 - math.cos(math.pi * k / n)) / 2
    label = lambda i, j: 1 + i + (n + 1) * j
    commands = []
    for j in range(n + 1):
        for i in range(n + 1):
            commands.append({"command": "mesh.node.add", "params": {"label": label(i, j), "xyz": [1000.0 * g(i) + s[0] * g(j), s[1] * g(j), 0.0]}})
    for j in range(n):
        for i in range(n):
            # clockwise seen from +z: the element normal e3 points down, so the shell's top (+e3)
            # surface is the plate's lower surface, the one the nodal stresses report
            nodes = [label(i, j), label(i, j + 1), label(i + 1, j + 1), label(i + 1, j)]
            commands.append({"command": "mesh.element.add", "params": {"label": 1 + i + n * j, "kind": "quad4", "nodes": nodes}})
    edges = sorted({label(i, 0) for i in range(n + 1)} | {label(i, n) for i in range(n + 1)}
                   | {label(0, j) for j in range(n + 1)} | {label(n, j) for j in range(n + 1)})
    return commands + [
        {"command": "model.set.create", "params": {"name": "plate", "elements": list(range(1, n * n + 1))}},
        {"command": "model.set.create", "params": {"name": "edges", "nodes": edges}},
        {"command": "model.set.create", "params": {"name": "A", "nodes": [label(0, 0)]}},
        {"command": "model.set.create", "params": {"name": "B", "nodes": [label(n, 0)]}},
        {"command": "model.material.create", "params": {"name": "Steel", "elastic": {"type": "isotropic", "young": 210000.0, "poisson": 0.3}}},
        {"command": "model.section.create", "params": {"name": "plate", "set": "plate", "material": "Steel", "kind": {"type": "shell", "thickness": 10.0}}},
        {"command": "model.boundary_condition.create", "params": {"name": "edges", "type": "displacement", "target": {"set": "edges"}, "values": [None, None, 0, None, None, None]}},
        {"command": "model.boundary_condition.create", "params": {"name": "A", "type": "displacement", "target": {"set": "A"}, "values": [0, 0, None, None, None, None]}},
        {"command": "model.boundary_condition.create", "params": {"name": "B", "type": "displacement", "target": {"set": "B"}, "values": [None, 0, None, None, None, None]}},
        # 0.7 kPa = 7e-4 MPa downward, spread over the 10 mm thickness as a load per volume
        {"command": "model.load.create", "params": {"name": "pressure", "type": "body_force", "set": "plate", "vector": [0, 0, -7e-5]}},
        {"command": "model.step.create", "params": {"name": "Load", "procedure": {"type": "static"}, "boundary_conditions": ["edges", "A", "B"], "loads": ["pressure"]}},
        {"command": "solve.static", "params": {}, "id": "solve"},
    ]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
print("[\n " + ",\n ".join(json.dumps(c) for c in le6(n)) + "\n]")
```

## Refinement

Every row is the listing above with the row's divisions (and, for the uniform rows, the grading
`g` replaced by k/n), solved and probed at E as the case does; the case mesh is the graded 64 × 64:

| cells | uniform grid | graded grid |
|---|---|---|
| 8 × 8 | 0.7465 | 0.7256 |
| 16 × 16 | 0.7751 | 0.8021 |
| 32 × 32 | 0.7952 | 0.8160 |
| 64 × 64 (this case: graded) | 0.8088 | 0.8197 |
| 128 × 128 | 0.8160 | 0.8210 |

The graded grid converges to about 0.821; the next refinement moves the case value by 0.15%.

## Towards the thin plate, and why the NAFEMS quantity is a known failure

The published target, 0.802, is a thin-plate value. Making the same model thinner, with the
pressure scaled with the thickness squared so that the thin-plate stress stays the same, moves σ₁
at E towards it, but slowly (graded grids, as listed above):

| thickness | span / thickness | 64 × 64 | 128 × 128 | 192 × 192 |
|---|---|---|---|---|
| 10 mm (the problem) | 100 | 0.8197 | 0.8210 | 0.8212 |
| 3 mm | 333 | 0.8096 | 0.8121 | |
| 1 mm | 1000 | 0.8025 | 0.8068 | 0.8077 |

The thinner the plate, the finer the grid it needs, so the 64 × 64 value at 1 mm (0.8025) is not
converged; the 1 mm plate is heading for about 0.809. The gap to 0.802 falls from 2.4% at 10 mm
to about 0.9% at 1 mm, roughly halving for each tenfold thinning. A slow approach is what a simply
supported plate with free rotations ("soft" support) shows: the transverse shear the MITC4 shell
keeps leaves boundary layers along the edges and at the obtuse corners that a thin-plate model has
not. That is why these runs are recorded here and not checked in CI. A third quantity at 1 mm
would pass only on an unconverged grid.

At the problem's thickness the shell converges to about 0.821, 2.4% above the thin-plate target,
and ESRD's converged thin-solid model lands at the same place (0.820 on two meshes; 0.822 with
triangles, p. 20). ESRD's guide also remarks that its thin-solid formulation can differ from the
plate model behind the analytical solution (p. 22). The case therefore keeps the NAFEMS target with
the tolerance its three figures support, 2%, marked `known_failure` until FieldCraft has a
thin-plate (Kirchhoff) element. It checks the shell against the shear-deformable reference at 1%:
0.8197 against 0.820.

`fixture.json` holds the passing quantity's value: the mesh is written by the recipe, so it is
bit-identical on every platform.

## Running it

```sh
cargo xtask oracle --case nafems-le6
cargo xtask oracle --bless nafems-le6   # only after the case passes
```
