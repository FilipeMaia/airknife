#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "manifold3d>=3.0",
#     "numpy>=1.24",
# ]
# ///
"""
Dual air-knife (pressure-wave) nozzles on a stationary, symmetric fork bolted to
the breadboard behind the plunger. Inspired by the EasyGrid spreading module
(Gemin et al., Nat. Methods 2026).

Layout (seen from above):

        breadboard
    ===================
            |              stem  (gas fitting on top, near the plate)
        +---+---+          crossbar at y = -fork_y
        |       |          two tines at x = +/-tine_x, outside the plunge path
        |   o   |          o = plunge axis (grid and tweezer pass through here)
        v       v          nozzles hang from the tine ends, pointing down/in

The grid face is at right angles to the breadboard, so the nozzles are left and
right of the grid as seen from the board. Everything is horizontal at one
height except the two nozzle drops; the part is mirror-symmetric about the grid
plane, including the gas path.

Usage:  uv run airknife_breadboard.py bb_dist=58 hole_u=-3.5 hole_z=12 stroke=45
Without uv:  pip install manifold3d numpy && python airknife_breadboard.py

Coordinates (mm): origin = grid centre in the SPREADING (up) position. +z up.
x = grid normal (light axis), y = along the grid face, toward the plunge axis
from the board. Breadboard surface at y = -bb_dist.

Breadboard hole reference (measure with the plunger UP):
  hole_u: horizontal offset of ANY breadboard hole from the plunge axis, along
          the board, positive to your RIGHT when you face the board.
  hole_z: height of that hole above the grid centre.
"""
import math
import sys

import numpy as np
from manifold3d import CrossSection, Manifold, OpType

# --------------------------------------------------------------- parameters --
P = dict(
    # breadboard and mounting
    bb_dist=86.5,         # board surface -> plunge axis (measured)
    shim=1.0,             # gap left behind the plate for washers (fine y-adjustment)
    pitch=25.0,           # hole pitch (25 metric, 25.4 for 1/4-20)
    hole_u=-7.5,          # any hole: offset from plunge axis along the board (measured)
    hole_z=0.0,           # same hole: height above grid centre, plunger up (measured)
    bolt_d=6.6,           # M6 clearance (6.9 for 1/4-20)
    adj_u=3.0,            # +/- horizontal adjustment of the oversized holes
    adj_z=3.0,            # +/- vertical adjustment
    head_clear=8.0,       # bolt heads at least this far from the stem
    plate_t=6.0,
    plate_margin=11.0,    # plate edge beyond hole centres
    gusset_t=5.0,
    # fork
    bar_w=8.0,            # width of stem, crossbar and tines
    bar_h=8.0,            # height of stem, crossbar and tines
    keep_r=10.0,          # radius around the plunge axis kept free (holder 8.5 + margin)
    port_from_plate=10.0, # fitting position along the stem, from the plate
    boss_d=12.0,          # flat seat around the M5 port
    port_pilot=4.2,
    port_depth=5.0,
    ch_d=2.5,
    drop_d=7.0,           # outer diameter of the drops down to the nozzles
    # nozzle = small air knife (same as before)
    theta=35.0,           # 35/13 instead of 30/11: opens the lip gap so the
    standoff=13.0,        # 8.4 mm lock ring passes with ~2 mm per side
    aim_dz=0.5,
    slot_w=5.0,
    slot_g=0.4,           # 0.6 for FDM/PLA
    land=2.0,
    conv=3.0,
    plen_g=2.5,
    plen_w=6.0,
    plen_h=3.0,
    lip=0.8,              # 1.0 for FDM/PLA
    wall=1.2,
    # moving parts for the collision check (plunger UP)
    stroke=67.0,          # grid travel from spreading position to bottom (measured)
    plunge_depth=5.0,
    holder_z0=101.5,      # holder bottom above grid centre
    holder_d=17.0,
    holder_len=45.5,
    rod_d=12.7,
    lock_z=75.0,          # lock-ring centre above grid centre  <-- CHECK
    lock_t=8.4,           # along the grid normal
    lock_w=10.0,          # along the grid face  <-- CHECK
    lock_h=12.0,          # <-- CHECK
    tip_above_center=0.8,
    tw_t5=2.3,
    tw_t10=3.3,
    tw_w=3.0,
    # output
    segments=64,
    out="airknife_breadboard.stl",
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


def teardrop_cs(d):
    """Channel section with a 45-degree point toward -z: prints upside down cleanly."""
    r = d / 2
    return (CrossSection.circle(r, SEG // 2)
            + CrossSection.square((0.05, 0.05), center=True).translate((0, -r * math.sqrt(2)))).hull()


def channel_y(x, y0, y1, z, d):
    """Horizontal teardrop channel along y from y0 to y1 (extended by d/2 each end)."""
    lo, hi = min(y0, y1) - d / 2, max(y0, y1) + d / 2
    return (Manifold.extrude(teardrop_cs(d), hi - lo).rotate((90, 0, 0))
            .translate((x, hi, z)))


def channel_x(x0, x1, y, z, d):
    lo, hi = min(x0, x1) - d / 2, max(x0, x1) + d / 2
    return (Manifold.extrude(teardrop_cs(d), hi - lo).rotate((90, 0, 0)).rotate((0, 0, -90))
            .translate((hi, y, z)))


# ----------------------------------------------------------------- nozzles ---
z_plen_top = g.land + g.conv + g.plen_h
Hn = z_plen_top + g.wall
th = math.radians(g.theta)
dirv = np.array([math.sin(th), 0.0, math.cos(th)])
aim = np.array([0.0, 0.0, g.aim_dz])
E = aim + g.standoff * dirv
I = E + Hn * dirv


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


def place_nozzle(m):
    return m.rotate((0, g.theta, 0)).translate(tuple(E))


# -------------------------------------------------------------------- fork ---
tine_x = g.keep_r + g.bar_w / 2                 # tines just outside the keep-out
fork_y = g.keep_r + g.bar_w / 2                 # crossbar just outside the keep-out
z_h = I[2] + (tine_x - I[0]) / math.tan(th)     # tine height: drop continues the nozzle axis
z_top = z_h + g.bar_h / 2
A = np.array([tine_x, 0.0, z_h])                # tine end, top of the drop
y_back = -g.bb_dist + g.shim
y_front = y_back + g.plate_t                    # plate face toward the plunger
y_port = y_front + g.port_from_plate
if y_port > -fork_y - g.boss_d:
    sys.exit("bb_dist too small: no room for the fitting on the stem")


def fork_outer():
    bh, bw = g.bar_h, g.bar_w
    stem = box(-bw / 2, bw / 2, y_front - 0.5, -fork_y + bw / 2, z_h - bh / 2, z_top)
    cross = box(-tine_x - bw / 2, tine_x + bw / 2, -fork_y - bw / 2, -fork_y + bw / 2,
                z_h - bh / 2, z_top)
    tines = both_sides(box(tine_x - bw / 2, tine_x + bw / 2, -fork_y, bw / 2,
                           z_h - bh / 2, z_top))
    boss = cyl(bh, g.boss_d, z_h - bh / 2).translate((0, y_port, 0))
    drops = both_sides(capsule(A, I, g.drop_d))
    return union([stem, cross, tines, boss, drops])


def fork_inner():
    d = g.ch_d
    return union([
        # M5 port straight down into the stem channel
        cyl(g.port_depth + 1, g.port_pilot, z_top - g.port_depth).translate((0, y_port, 0)),
        cyl(z_top - z_h, d, z_h).translate((0, y_port, 0)),
        # stem -> T -> crossbar -> tines -> drops (all mirror-symmetric)
        channel_y(0, y_port, -fork_y, z_h, d),
        channel_x(-tine_x, tine_x, -fork_y, z_h, d),
        both_sides(channel_y(tine_x, -fork_y, 0, z_h, d)),
        both_sides(capsule(A, I, d)),
    ])


# ---------------------------------------------------------- plate + gusset ---
def pick_holes():
    clear = g.bar_w / 2 + g.head_clear
    k1 = math.floor((-clear - g.hole_u) / g.pitch)
    k2 = math.ceil((clear - g.hole_u) / g.pitch)
    c1, c2 = g.hole_u + k1 * g.pitch, g.hole_u + k2 * g.pitch
    m = math.floor((z_top - g.bar_h - g.plate_margin - g.hole_z) / g.pitch)
    return c1, c2, g.hole_z + m * g.pitch


c1, c2, row = pick_holes()
half_w = max(-c1, c2) + g.plate_margin            # symmetric plate
plate_z0 = row - g.plate_margin


def hole_shape():
    # u = along the board (+x seen from the plunger is -u; the plate is symmetric anyway)
    rr = CrossSection.square((max(2 * g.adj_u, 0.01), max(2 * g.adj_z, 0.01)), center=True).offset(
        g.bolt_d / 2, circular_segments=SEG)
    tip = CrossSection.square((0.05, 0.05), center=True).translate(
        (0, -(g.adj_z + g.bolt_d / 2 * math.sqrt(2))))
    return Manifold.extrude((rr + tip).hull(), g.plate_t + 2).rotate((90, 0, 0))


def plate():
    p = box(-half_w, half_w, y_back, y_front, plate_z0, z_top)
    # facing the board from the plunger, "right" is -x
    holes = union([hole_shape().translate((-c, y_front + 1, row)) for c in (c1, c2)])
    zb = z_h - g.bar_h / 2
    gl = min(0.7 * (-fork_y - y_front), zb - plate_z0)
    gusset = Manifold.batch_hull([
        box(-g.gusset_t / 2, g.gusset_t / 2, y_front - 0.5, y_front, zb - gl, zb + 0.1),
        box(-g.gusset_t / 2, g.gusset_t / 2, y_front - 0.5, y_front + gl, zb, zb + 0.1),
    ])
    return (p - holes) + gusset


# ---------------------------------------------------------------- assemble ---
outer = union([plate(), fork_outer(), both_sides(place_nozzle(nozzle_outer()))])
inner = fork_inner() + both_sides(place_nozzle(nozzle_inner()))
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
holder = cyl(g.holder_len, g.holder_d, g.holder_z0)
rod = cyl(60, g.rod_d, g.holder_z0 + g.holder_len)
lock = cube((g.lock_t, g.lock_w, g.lock_h), (0, 0, g.lock_z))

lip_a = g.slot_g / 2 + g.lip
lip_gap = 2 * (E[0] - lip_a * math.cos(th))
bb = part.bounding_box()
print(f"Fork height: channels {z_h:.1f} mm, top face {z_top:.1f} mm above grid centre")
print(f"Tines at x = +/-{tine_x:.1f} mm, crossbar at y = -{fork_y:.1f} mm "
      f"(keep-out radius {g.keep_r} mm around the plunge axis)")
print(f"Plate: bb_dist = {g.bb_dist} mm, {g.shim} mm left for shim washers")
print(f"Bolt holes: board columns u = {c1:.1f} and {c2:.1f} mm (right of axis = +), "
      f"row z = {row:.1f} mm; adjust +/-{g.adj_u}/{g.adj_z} mm")
print(f"Free gap between nozzle lips: {lip_gap:.2f} mm (everything passing down must be thinner)")
print(f"Lowest point: {bb[2]:.1f} mm above grid centre; ethane surface "
      f"{g.stroke - g.plunge_depth:.1f} mm below it")
print(f"Moving parts swept over a {g.stroke} mm stroke:")
for name, shp in [("tweezer (lower 10 mm)", tweezer), ("lock ring", lock),
                  ("holder", holder), ("rod", rod)]:
    sw = sweep(shp)
    gap = part.min_gap(sw, 30.0)
    hit = (part ^ sw).volume() > 1e-6
    print(f"  {name:22s} min gap {gap:6.2f} mm" + ("  <-- COLLISION" if hit else ""))
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
        f.write(b"airknife breadboard fork".ljust(80, b" "))
        f.write(np.uint32(len(t)).tobytes())
        f.write(rec.tobytes())
    print(f"Wrote {path} ({len(t)} triangles). Print upside down, flat top face on the bed.")


write_stl(part, g.out)
