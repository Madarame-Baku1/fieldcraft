// The free nodes' part of fieldcraft_mesh's coloured Winslow sweeps (project/smooth.rs), in
// single precision. One thread per node of the colour class: Winslow's update from the node's
// stencil (or the mean of its edge neighbours), then the guarded step — the full move, else half,
// else a quarter, the first that keeps the smallest scaled Jacobian of the elements around the
// node at or above the lower of its value before and the floor, and their largest edge ratio at
// or below the higher of its value before and the ceiling. No two nodes of a class share an
// element, so a thread reads only nodes no other thread of the dispatch writes, and each moves
// its own node in place.
//
// Positions are relative to the mesh's centre (the host adds it back).

struct Params {
    // First node of the class in `rules` and `out`, and its node count.
    start: u32,
    count: u32,
    floor: f32,
    min_move: f32,
    // Threads per row of the dispatch grid (workgroups in x times 64).
    row: u32,
    // Where the element lists start in `around` (after the node count + 1 offsets).
    around_base: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> prm: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
// Eight corner nodes per element, in hex8 order.
@group(0) @binding(2) var<storage, read> hexes: array<u32>;
// Per node the offsets of its elements, then the element lists.
@group(0) @binding(3) var<storage, read> around: array<u32>;
// Per class node: node, rule (0 Winslow, 1 mean), then for Winslow the six edge neighbours
// nb[d][s], the twelve face diagonals diag[pair][sa][sb] and the three control functions (as
// bits); for a mean the offset and length of its neighbours in `means`.
@group(0) @binding(4) var<storage, read> rules: array<u32>;
@group(0) @binding(5) var<storage, read> means: array<u32>;
// Per class node: where it is after the sweep, and w = 0 (stayed), 1 (moved), 2 (refused).
@group(0) @binding(6) var<storage, read_write> out: array<vec4<f32>>;
// Nodes the host moved: where to, and which.
@group(0) @binding(7) var<storage, read> upd: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> upd_node: array<u32>;

const STRIDE: u32 = 24u;
const RATIO_CEILING: f32 = 5.0;
const BIG: f32 = 3.0e38;

// The corners of the unit cube in hex8 order.
const CORNERS = array<vec3<u32>, 8>(
    vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u), vec3<u32>(1u, 1u, 0u), vec3<u32>(0u, 1u, 0u),
    vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u), vec3<u32>(1u, 1u, 1u), vec3<u32>(0u, 1u, 1u),
);

// Where the scaled Jacobian is also measured inside: the centre and the six face centres.
const INSIDE = array<vec3<f32>, 7>(
    vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(0.5, 0.5, 0.0), vec3<f32>(0.5, 0.5, 1.0), vec3<f32>(0.5, 0.0, 0.5),
    vec3<f32>(0.5, 1.0, 0.5), vec3<f32>(0.0, 0.5, 0.5), vec3<f32>(1.0, 0.5, 0.5),
);

fn corner_at(b: vec3<u32>) -> u32 {
    var k = select(select(0u, 1u, b.x == 1u), select(3u, 2u, b.x == 1u), b.y == 1u);
    return k + 4u * b.z;
}

fn flip(b: vec3<u32>, d: u32) -> vec3<u32> {
    var f = b;
    f[d] = 1u - f[d];
    return f;
}

fn normalized(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let l = length(a) * length(b) * length(c);
    if (l > 0.0 && l < BIG) {
        return dot(a, cross(b, c)) / l;
    }
    return -1.0;
}

struct Local {
    sj: f32,
    ratio: f32,
}

// The smallest scaled Jacobian (corners, centre and face centres) and the edge ratio of one
// element (project/smooth.rs: hex_scaled_jacobian, hex_edge_ratio).
fn element(x: array<vec3<f32>, 8>) -> Local {
    var c = x;
    var corners = CORNERS;
    var inside = INSIDE;
    var m = BIG;
    for (var k = 0u; k < 8u; k++) {
        let bits = corners[k];
        var e: array<vec3<f32>, 3>;
        for (var d = 0u; d < 3u; d++) {
            var v = c[corner_at(flip(bits, d))] - c[k];
            if (bits[d] == 1u) {
                v = -v;
            }
            e[d] = v;
        }
        m = min(m, normalized(e[0], e[1], e[2]));
    }
    for (var q = 0u; q < 7u; q++) {
        let u = inside[q];
        var e: array<vec3<f32>, 3>;
        for (var d = 0u; d < 3u; d++) {
            var s = vec3<f32>(0.0);
            for (var k = 0u; k < 8u; k++) {
                let bits = corners[k];
                var w = 1.0;
                for (var j = 0u; j < 3u; j++) {
                    if (j == d) {
                        w *= select(-1.0, 1.0, bits[j] == 1u);
                    } else {
                        w *= select(1.0 - u[j], u[j], bits[j] == 1u);
                    }
                }
                s += c[k] * w;
            }
            e[d] = s;
        }
        m = min(m, normalized(e[0], e[1], e[2]));
    }
    var lo = BIG;
    var hi = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let bits = corners[k];
        for (var d = 0u; d < 3u; d++) {
            if (bits[d] == 0u) {
                let l = distance(c[k], c[corner_at(flip(bits, d))]);
                lo = min(lo, l);
                hi = max(hi, l);
            }
        }
    }
    var ratio = BIG;
    if (lo > 0.0) {
        ratio = hi / lo;
    }
    return Local(m, ratio);
}

// The quality of the elements around node `v` with `v` at `p`.
fn quality(v: u32, p: vec3<f32>) -> Local {
    var l = Local(BIG, 1.0);
    let first = around[v];
    let end = around[v + 1u];
    for (var j = first; j < end; j++) {
        let h = around[prm.around_base + j];
        var x: array<vec3<f32>, 8>;
        for (var k = 0u; k < 8u; k++) {
            let w = hexes[8u * h + k];
            x[k] = select(pos[w].xyz, p, w == v);
        }
        let e = element(x);
        l.sj = min(l.sj, e.sj);
        l.ratio = max(l.ratio, e.ratio);
    }
    return l;
}

fn at(w: u32) -> vec3<f32> {
    return pos[w].xyz;
}

struct Target {
    q: vec3<f32>,
    ok: bool,
}

// Winslow's update of a node with a structured stencil (project/smooth.rs: winslow).
fn winslow(r: u32) -> Target {
    var xi: array<vec3<f32>, 3>;
    for (var d = 0u; d < 3u; d++) {
        xi[d] = 0.5 * (at(rules[r + 2u + 2u * d + 1u]) - at(rules[r + 2u + 2u * d]));
    }
    let g11 = dot(xi[0], xi[0]);
    let g22 = dot(xi[1], xi[1]);
    let g33 = dot(xi[2], xi[2]);
    let g12 = dot(xi[0], xi[1]);
    let g13 = dot(xi[0], xi[2]);
    let g23 = dot(xi[1], xi[2]);
    var cof_diag = array<f32, 3>(g22 * g33 - g23 * g23, g11 * g33 - g13 * g13, g11 * g22 - g12 * g12);
    var cof_off = array<f32, 3>(g13 * g23 - g12 * g33, g12 * g23 - g13 * g22, g13 * g12 - g11 * g23);
    let den = 2.0 * (cof_diag[0] + cof_diag[1] + cof_diag[2]);
    if (!(den > 0.0 && den < BIG)) {
        return Target(vec3<f32>(0.0), false);
    }
    var num = vec3<f32>(0.0);
    for (var d = 0u; d < 3u; d++) {
        let m = at(rules[r + 2u + 2u * d]);
        let pl = at(rules[r + 2u + 2u * d + 1u]);
        let control = bitcast<f32>(rules[r + 20u + d]);
        num += (m + pl) * cof_diag[d];
        num += (pl - m) * (0.5 * cof_diag[d] * control);
    }
    for (var q = 0u; q < 3u; q++) {
        let dg = r + 8u + 4u * q;
        let cross_d = 0.25 * ((at(rules[dg + 3u]) + at(rules[dg])) - (at(rules[dg + 2u]) + at(rules[dg + 1u])));
        num += cross_d * (2.0 * cof_off[q]);
    }
    let out = num / den;
    let ok = all(abs(out) < vec3<f32>(BIG));
    return Target(out, ok);
}

// The mean of a node's edge neighbours.
fn mean(r: u32) -> Target {
    let first = rules[r + 2u];
    let len = rules[r + 3u];
    if (len == 0u) {
        return Target(vec3<f32>(0.0), false);
    }
    var s = vec3<f32>(0.0);
    for (var j = 0u; j < len; j++) {
        s += at(means[first + j]);
    }
    return Target(s * (1.0 / f32(len)), true);
}

@compute @workgroup_size(64)
fn sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * prm.row;
    if (i >= prm.count) {
        return;
    }
    let slot = prm.start + i;
    let r = slot * STRIDE;
    let v = rules[r];
    let p = at(v);
    var t: Target;
    if (rules[r + 1u] == 0u) {
        t = winslow(r);
    } else {
        t = mean(r);
    }
    if (!t.ok || distance(t.q, p) <= prm.min_move) {
        out[slot] = vec4<f32>(p, 0.0);
        return;
    }
    let before = quality(v, p);
    let floor = min(before.sj, prm.floor);
    let ceiling = max(before.ratio, RATIO_CEILING);
    var steps = array<f32, 3>(1.0, 0.5, 0.25);
    for (var s = 0u; s < 3u; s++) {
        var to = t.q;
        if (s > 0u) {
            to = p + (t.q - p) * steps[s];
        }
        let after = quality(v, to);
        if (after.sj >= floor && after.ratio <= ceiling) {
            pos[v] = vec4<f32>(to, 0.0);
            out[slot] = vec4<f32>(to, 1.0);
            return;
        }
    }
    out[slot] = vec4<f32>(p, 2.0);
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * prm.row;
    if (i >= prm.count) {
        return;
    }
    pos[upd_node[i]] = vec4<f32>(upd[i].xyz, 0.0);
}
