#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "manifold3d>=3.0",
#     "numpy>=1.24",
# ]
# ///
"""
Dual air-knife (pressure-wave) nozzles that blow HORIZONTALLY across the grid,
away from the breadboard. Stationary, mirror-symmetric, bolted to the
breadboard behind the plunger. Inspired by the EasyGrid spreading module
(Gemin et al., Nat. Methods 2026).

Seen from above (breadboard at the bottom):

          +y  (jets blow this way, liquid leaves the far edge of the grid)
           ^
     \\    |    /      <- the two jets graze the two grid faces
      \\   |   /
       [] grid []     <- nozzles stay behind the grid, out of the plunge path
        \\  |  /
         \\ | /       <- symmetric Y-split
           |
           |   stem (gas fitting on top, near the plate)
    =================  breadboard

Everything lies in one horizontal plane at grid height. Nothing blows down
into the ethane / LN2, and nothing sits in the plunge path: the tweezer, lock
ring and holder drop past the front of the nozzles.

Usage:  uv run airknife_horizontal.py slot_g=0.6 lip=1.0      (PLA settings)
Without uv:  pip install manifold3d numpy && python airknife_horizontal.py

Coordinates (mm): origin = grid centre in the SPREADING (up) position. +z up.
x = grid normal (light axis), y = along the grid face, away from the board.
Breadboard surface at y = -bb_dist.

Breadboard hole reference (plunger UP): hole_u = horizontal offset of any hole
from the plunge axis, positive to your RIGHT when facing the board; hole_z =
its height above the grid centre.
"""
import math
import sys

import numpy as np
from manifold3d import CrossSection, Manifold, OpType

# --------------------------------------------------------------- parameters --
P = dict(
    # breadboard and mounting (measured)
    bb_dist=86.5,
    shim=1.0,             # gap left behind the plate for washers (fine y-adjustment)
    pitch=25.0,
    hole_u=-7.5,
    hole_z=0.0,
    bolt_d=6.6,           # M6 clearance (6.9 for 1/4-20)
    adj_u=3.0,            # +/- horizontal adjustment of the oversized holes
    adj_z=3.0,            # +/- vertical adjustment
    head_clear=8.0,       # bolt heads at least this far from the stem
    plate_t=6.0,
    plate_margin=11.0,
    # stem, split and gussets
    bar_w=10.0,           # stem width (x)
    bar_h=10.0,           # stem height (z)
    gusset=10.0,          # 45-degree gussets above and below the stem root
    washer_r=9.0,         # bolt washer radius (M6 large washer, 18 mm OD)
    hole_layout="column", # "column": two holes above/below the stem (25 mm pitch checked);
                          # "row": left/right of the stem, level with the grid;
                          # "auto": whichever gives the smaller plate
    gusset_t=4.0,
    port_from_plate=24.0, # fitting position along the stem, clear of the gusset
    fitting_r=8.0,        # radius kept free above the port (QSM hex + spanner)
    boss_d=12.0,
    port_pilot=4.2,
    port_depth=5.0,
    ch_d=2.5,
    arm_d=7.0,
    # nozzles = small air knives, slot vertical, jets horizontal
    theta=30.0,           # jet angle to the grid face (in the horizontal plane)
    standoff=11.0,        # exit -> impact point on the grid face
    aim_y=-1.5,           # impact point along the face (-1.5 = edge nearest the board)
    aim_dz=0.0,           # slot centre height relative to grid centre
    slot_w=5.0,           # slot length, now VERTICAL (grid is 3.05 mm)
    slot_g=0.4,           # 0.6 for FDM/PLA
    land=2.0,
    conv=3.0,
    plen_g=2.5,
    plen_w=6.0,
    plen_h=3.0,
    lip=0.8,              # 1.0 for FDM/PLA
    wall=1.2,
    # moving parts for the collision check (plunger UP)
    stroke=67.0,
    plunge_depth=5.0,
    holder_z0=101.5,
    holder_d=17.0,
    holder_len=45.5,
    rod_d=12.7,
    lock_z=75.0,          # lock-ring centre above grid centre  <-- CHECK
    lock_t=8.4,           # along the grid normal
    lock_w=10.0,          # along the grid face  <-- CHECK
    lock_h=12.0,
    tip_above_center=0.8,
    tw_t5=2.3,
    tw_t10=3.3,
    tw_w=3.0,             # tweezer blade width near the tip (along the face)  <-- CHECK
    light_r=5.0,          # radius of the light beam at the grid (check only)
    # output
    segments=64,
    out="airknife_horizontal.stl",
)

for arg in sys.argv[1:]:
    k, v = arg.split("=", 1)
    if k not in P:
        sys.exit(f"Unknown parameter: {k}")
    P[k] = v if isinstance(P[k], str) else type(P[k])(float(v))

g = type("Params", (), P)
SEG = int(g.segments)


# ------------------------------------------------------------------ helpers --
def cube(size, center_xyz):
    return Manifold.cube(size, center=True).translate(center_xyz)


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def cyl(h, d, z0=0.0, seg=SEG):
    return Manifold.cylinder(h, d / 2, circular_segments=seg).translate((0, 0, z0))


def sphere(d, at):
    return Manifold.sphere(d / 2, circular_segments=SEG // 2).translate(tuple(at))


def capsule(a, b, d):
    return Manifold.batch_hull([sphere(d, a), sphere(d, b)])


def both_sides(m):
    return m + m.mirror((1, 0, 0))


def union(parts):
    return Manifold.batch_boolean(parts, OpType.Add) if len(parts) > 1 else parts[0]


# ----------------------------------------------------------------- nozzles ---
th = math.radians(g.theta)
jet = np.array([-math.sin(th), math.cos(th), 0.0])      # +x nozzle blows toward -x, +y
back = -jet                                             # nozzle body extends this way
gapn = np.array([math.cos(th), math.sin(th), 0.0])      # slot gap direction
aim = np.array([0.0, g.aim_y, g.aim_dz])
E = aim + g.standoff * back                             # exit centre (+x nozzle)
z_plen_top = g.land + g.conv + g.plen_h
Hn = z_plen_top + g.wall
I = E + Hn * back                                       # inlet
S = np.array([0.0, I[1] - abs(I[0]) - 1.0, g.aim_dz])   # split: branches <= 45 deg off the stem
# local nozzle frame: x_l -> gap direction, y_l -> world z (slot length), z_l -> back
M = np.column_stack([gapn, [0, 0, 1], back, E])


def nozzle_outer():
    tip = cube((g.slot_g + 2 * g.lip, g.slot_w + 2 * g.lip, 0.1), (0, 0, 0.05))
    hblk = Hn - g.land - g.conv
    blk = cube((g.plen_g + 2 * g.wall, g.plen_w + 2 * g.wall, hblk),
               (0, 0, g.land + g.conv + hblk / 2))
    return Manifold.batch_hull([tip, blk])


def nozzle_inner():
    slot = cube((g.slot_g, g.slot_w, g.land + 1.01), (0, 0, (g.land - 1) / 2))
    converge = Manifold.batch_hull([
        cube((g.slot_g, g.slot_w, 0.02), (0, 0, g.land)),
        cube((g.plen_g, g.plen_w, 0.02), (0, 0, g.land + g.conv)),
    ])
    plenum = cube((g.plen_g, g.plen_w, g.plen_h), (0, 0, g.land + g.conv + g.plen_h / 2))
    inlet = cyl(Hn - z_plen_top + 1, g.ch_d, z_plen_top - 0.5)
    return union([slot, converge, plenum, inlet])


def place(m):
    return m.transform(M)


# ------------------------------------------------------- plate, stem, port ---
y_back = -g.bb_dist + g.shim
y_front = y_back + g.plate_t
y_port = y_front + g.port_from_plate
zc = g.aim_dz
if y_port + g.boss_d / 2 > S[1] - 2:
    sys.exit("bb_dist too small for the stem")


def pick_holes():
    """Pick two breadboard holes: a pair left/right of the stem in one row, or a
    pair above/below the stem in one column, whichever gives the smaller plate.
    Washers must clear the stem, the gussets and the fitting/tube above it."""
    def col(k):
        return g.hole_u + k * g.pitch

    def rowz(m):
        return g.hole_z + m * g.pitch

    opts = []
    # (a) same row, columns either side of the stem
    cu = g.bar_w / 2 + g.washer_r + 1
    k1 = math.floor((-cu - g.hole_u) / g.pitch)
    k2 = math.ceil((cu - g.hole_u) / g.pitch)
    r = rowz(round((zc - g.hole_z) / g.pitch))
    if g.hole_layout in ("row", "auto"):
        opts.append([(col(k1), r), (col(k2), r)])
    # (b) same column, rows above and below the stem; column >= 5 mm off-centre
    #     so the hex key clears the gas tube rising from the fitting
    cz = g.bar_h / 2 + g.gusset + g.washer_r
    m1 = math.floor((zc - cz - g.hole_z) / g.pitch)
    m2 = math.ceil((zc + cz - g.hole_z) / g.pitch)
    k = min(range(-3, 4), key=lambda k: abs(col(k)) if abs(col(k)) >= 5 else 1e9)
    if g.hole_layout in ("column", "auto"):
        opts.append([(col(k), rowz(m1)), (col(k), rowz(m2))])
    if not opts:
        sys.exit('hole_layout must be "row", "column" or "auto"')

    def extent(h):
        hw = max(abs(u) for u, _ in h) + g.plate_margin
        z0 = min(min(z for _, z in h) - g.plate_margin, zc - g.bar_h / 2 - g.gusset - 2)
        z1 = max(max(z for _, z in h) + g.plate_margin, zc + g.bar_h / 2 + g.gusset + 2)
        return hw, z0, z1
    best = min(opts, key=lambda h: 2 * extent(h)[0] * (extent(h)[2] - extent(h)[1]))
    return best, extent(best)


holes, (half_w, pz0, pz1) = pick_holes()


def hole_shape():
    rr = CrossSection.square((max(2 * g.adj_u, 0.01), max(2 * g.adj_z, 0.01)), center=True).offset(
        g.bolt_d / 2, circular_segments=SEG)
    # extrude along y (the plate is printed flat, so the holes print vertical)
    return Manifold.extrude(rr, g.plate_t + 2).rotate((-90, 0, 0))


def body_outer():
    plate = box(-half_w, half_w, y_back, y_front, pz0, pz1)
    stem = box(-g.bar_w / 2, g.bar_w / 2, y_front - 0.5, S[1], zc - g.bar_h / 2, zc + g.bar_h / 2)
    boss = cyl(g.bar_h, g.boss_d, zc - g.bar_h / 2).translate((0, y_port, 0))
    node = sphere(g.arm_d + 1, S)
    gl = g.gusset
    gussets = []
    for sx, sz in [(0, 1), (0, -1)]:
        # triangle in the plane of the stem axis, 45 deg from the plate face
        ex = sx * (g.bar_w / 2 + gl)
        ez = sz * (g.bar_h / 2 + gl)
        t = g.gusset_t / 2
        if sx:
            a = box(min(0, ex), max(0, ex), y_front - 0.5, y_front, zc - t, zc + t)
            b = box(-0.01, 0.01, y_front - 0.5, y_front + gl + g.bar_w / 2, zc - t, zc + t)
        else:
            a = box(-t, t, y_front - 0.5, y_front, zc + min(0, ez), zc + max(0, ez))
            b = box(-t, t, y_front - 0.5, y_front + gl + g.bar_h / 2, zc - 0.01, zc + 0.01)
        gussets.append(Manifold.batch_hull([a, b]))
    return union([plate, stem, boss, node] + gussets) - union(
        [hole_shape().translate((-u, y_back - 1, z)) for u, z in holes])


def body_inner():
    d = g.ch_d
    # M5 port from the top of the boss. Plain round bore (it prints horizontally;
    # the small sag at its roof is removed by the tap) so the fitting's seal
    # sits on an unbroken flat face.
    port = cyl(g.port_depth + 1, g.port_pilot, zc + g.bar_h / 2 - g.port_depth).translate((0, y_port, 0))
    down = Manifold.batch_hull([
        cyl(g.bar_h / 2 + 0.5, d, zc),
        cube((0.05, 0.05, g.bar_h / 2 + 0.5), (0, d / 2 * math.sqrt(2), zc + g.bar_h / 4 + 0.25)),
    ]).translate((0, y_port, 0))
    stem_ch = capsule((0, y_port, zc), S, d)                 # vertical when printed
    return union([port, down, stem_ch, sphere(d, S)])


# ---------------------------------------------------------------- assemble ---
arm_outer = capsule(I, S, g.arm_d)
arm_inner = capsule(I, S, g.ch_d)
outer = union([body_outer(), both_sides(place(nozzle_outer()) + arm_outer)])
inner = body_inner() + both_sides(place(nozzle_inner()) + arm_inner)
part = outer - inner
if part.status().name != "NoError" or part.is_empty():
    sys.exit(f"Geometry error: {part.status()}")


# ------------------------------------------------- collision check (moving) --
def sweep(shape):
    return Manifold.batch_hull([shape, shape.translate((0, 0, -g.stroke))])


slope = (g.tw_t10 - g.tw_t5) / 5.0
t0 = g.tw_t5 - 5 * slope
zt = g.tip_above_center
tweezer = Manifold.batch_hull([
    cube((max(t0, 0.2), g.tw_w, 0.01), (0, 0, zt)),
    cube((t0 + slope * 10, g.tw_w, 0.01), (0, 0, zt + 10)),
])
grid = cyl(0.1, 3.05, -0.05).rotate((0, 90, 0))
holder = cyl(g.holder_len, g.holder_d, g.holder_z0)
rod = cyl(60, g.rod_d, g.holder_z0 + g.holder_len)
lock = cube((g.lock_t, g.lock_w, g.lock_h), (0, 0, g.lock_z))
beam = cyl(200, 2 * g.light_r, -100).rotate((0, 90, 0))

lip_a = g.slot_g / 2 + g.lip
inner_lip = E - lip_a * gapn
print(f"Nozzle exits at x = +/-{E[0]:.2f}, y = {E[1]:.2f} mm; jets {g.theta} deg to the face, "
      f"blowing toward +y (away from the board)")
print(f"Closest nozzle lip: x = {inner_lip[0]:.2f}, y = {inner_lip[1]:.2f} mm "
      f"(the plunge path is in front of it, at y > {-max(g.lock_w, g.tw_w, 3.05) / 2:.1f})")
print(f"Split {-S[1]:.1f} mm behind the grid; plate face {-y_front:.1f} mm behind")
print("Bolt holes (u = right of axis when facing the board, z = above grid centre): "
      + ", ".join(f"(u {u:.1f}, z {z:.1f})" for u, z in holes)
      + f"; adjust +/-{g.adj_u}/{g.adj_z} mm")
print(f"Plate: {2 * half_w:.0f} mm wide x {pz1 - pz0:.0f} mm tall")
top = zc + g.bar_h / 2
above = part ^ box(-300, 300, -300, 300, top + 0.2, top + 300)
axis = cyl(30, 0.1, top + 0.2).translate((0, y_port, 0))
seat_free = above.is_empty() or above.volume() < 1e-6
r_free = 99.0 if seat_free else above.min_gap(axis, 50.0)
print(f"Fitting seat: flat, {g.boss_d:.0f} mm across; nearest part above the seat is "
      f"{r_free:.1f} mm from the port axis" + ("  <-- TOO CLOSE" if r_free < g.fitting_r else ""))
print(f"Moving parts swept over a {g.stroke} mm stroke:")
for name, shp in [("grid", grid), ("tweezer (lower 10 mm)", tweezer), ("lock ring", lock),
                  ("holder", holder), ("rod", rod)]:
    sw = sweep(shp)
    gap = part.min_gap(sw, 40.0)
    hit = (part ^ sw).volume() > 1e-6
    print(f"  {name:22s} min gap {gap:6.2f} mm" + ("  <-- COLLISION" if hit else ""))
print(f"Light beam along x (radius {g.light_r} mm): "
      + ("BLOCKED" if (part ^ beam).volume() > 1e-6 else f"clear, gap {part.min_gap(beam, 40):.1f} mm"))
print(f"Part volume: {part.volume() / 1000:.1f} cm^3 (~{part.volume() / 1000 * 1.24:.0f} g PLA)")


def write_stl(m, path):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
    t = np.asarray(mesh.tri_verts)
    tri = v[t]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as f:
        f.write(b"airknife horizontal".ljust(80, b" "))
        f.write(np.uint32(len(t)).tobytes())
        f.write(rec.tobytes())
    print(f"Wrote {path} ({len(t)} triangles). Print with the plate's back face "
          f"(the side against the breadboard) on the bed.")


write_stl(part, g.out)
