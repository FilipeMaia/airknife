#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "manifold3d>=3.0",
#     "numpy>=1.24",
# ]
# ///
"""
Dual air-knife (pressure-wave) manifold for a plunger tweezer holder.
Inspired by the EasyGrid spreading module (Gemin et al., Nat. Methods 2026).

Version 2:
  * Through-bore: the part slides UP along the holder and the 12.7 mm rod, so it
    can be lifted out of the way to take the tweezer out from below.
  * The axial reference is now the TOP face of the holder: the step between the
    holder bore and the rod bore sits on it (measure holder_len carefully).
  * Indexing/retention by an M4 ball spring plunger that clicks into the holder
    slot and rolls out when the part is pulled up (printed spring as option).
  * Extended sleeve over the rod replaces the green alignment collar.

Usage (uv installs the dependencies automatically, in an isolated environment):
               uv run airknife_manifold.py                      -> airknife_manifold.stl
               uv run airknife_manifold.py slot_g=0.6 lip=1.0          (PLA settings)
  or, after chmod +x:  ./airknife_manifold.py
Without uv:    pip install manifold3d numpy && python airknife_manifold.py

Coordinates (mm): origin = centre of the BOTTOM face of the 17 mm tweezer holder
(part seated). +z up, grid centre at z = -grid_z. Grid plane = y-z plane (x = 0);
+x = grid normal (light axis). The nozzles are mirror images across the grid plane.
"""
import math
import sys

import numpy as np
from manifold3d import CrossSection, Manifold, OpType

# --------------------------------------------------------------- parameters --
P = dict(
    # holder / rod / plunger
    holder_d=17.0,        # tweezer-holder diameter
    fit_clear=0.35,       # diametral clearance on the holder bore (PLA on A1)
    holder_len=45.5,      # holder bottom face -> holder top face (rod step), measured
    rod_d=12.7,           # support rod above the holder
    rod_clear=0.35,       # diametral clearance on the rod bore (same feel as holder bore)
    rod_cover=30.0,       # how far the sleeve extends up the rod
    upper_r=11.0,         # outer radius of the sleeve above the manifold
    collar_h=14.0,        # height of the full-radius section above holder bottom
    grid_z=101.5,         # holder bottom face -> grid centre (measured)
    chamfer=0.8,          # lead-in chamfers on both bore ends
    # holder slot (3 mm wide, open at the bottom, 10 mm long)
    slot_angle=90.0,      # slot direction, deg from +x; 90 = parallel to grid face
    slot_len=10.0,
    # detent that clicks into the slot opening (replaces the fixed keys)
    detent=2,             # 0 = none, 1 = printed spring band, 2 = M4 ball spring plunger
    detent_z=5.0,         # detent height above holder bottom (inside the slot)
    plunger_pilot=3.3,    # M4 tap drill for the ball spring plunger
    plunger_ball=2.5,     # ball diameter (must fit the 3 mm slot)
    plunger_proud=0.8,    # ball stroke / protrusion into the bore
    # printed spring band (detent=1 only; slicers may flag it as a cantilever)
    bump_h=0.5,           # how far the bump pokes into the bore
    bump_w=2.6,           # bump width across the slot (slot is 3 mm)
    bump_r=1.5,           # bump rounding radius (larger = softer click)
    tab_t=1.0,            # radial thickness of the spring band (stiffer = thicker)
    tab_h=3.5,            # height of the spring band
    tab_arc=160.0,        # free span of the band, deg of arc (longer = softer)
    tab_gap=0.8,          # clearance around the band (> bump_h)
    # optional radial M3 set screw (locks the part; must be loosened to lift)
    clamp=0,
    clamp_pilot=2.5,
    # gas inlet: Festo QSM-M5 at the top; the feed runs down a slim rib at
    # radius ring_r and only jogs out to port_y near the top
    port_y=19.0,
    port_pilot=4.2,
    port_depth=5.5,
    boss_d=12.0,          # flat seat for the fitting at the top
    col_d=7.0,            # width of the feed rib
    # manifold and channels
    body_r=15.0,
    manif_h=8.0,
    ch_d=2.5,
    ring_r=12.0,          # U-split radius = arm x position
    arm_d=7.0,
    arm_rise=5.0,
    # nozzle = small air knife
    theta=30.0,
    standoff=11.0,
    aim_dz=0.5,
    slot_w=5.0,
    slot_g=0.4,           # use 0.6 for FDM/PLA
    land=2.0,
    conv=3.0,
    plen_g=2.5,
    plen_w=6.0,
    plen_h=3.0,
    lip=0.8,              # use 1.0 for FDM/PLA
    wall=1.2,
    # clearance checks (not part of the geometry)
    plunge_depth=5.0,
    tip_above_center=0.8,
    tw_t5=2.3,
    tw_t10=3.3,
    tw_w=3.0,
    lock_ring_t=8.4,
    # output
    segments=64,
    out="airknife_manifold.stl",
)

for arg in sys.argv[1:]:                       # simple key=value overrides
    k, v = arg.split("=", 1)
    if k not in P:
        sys.exit(f"Unknown parameter: {k}")
    P[k] = v if isinstance(P[k], str) else type(P[k])(float(v))

g = type("Params", (), P)
SEG = int(g.segments)


# ------------------------------------------------------------------ helpers --
def cube(size, center_xyz):
    return Manifold.cube(size, center=True).translate(center_xyz)


def cyl(h, d, z0=0.0, seg=SEG):
    return Manifold.cylinder(h, d / 2, circular_segments=seg).translate((0, 0, z0))


def cone(h, d0, d1, z0=0.0, seg=SEG):
    return Manifold.cylinder(h, d0 / 2, d1 / 2, circular_segments=seg).translate((0, 0, z0))


def sphere(d, at):
    return Manifold.sphere(d / 2, circular_segments=SEG // 2).translate(at)


def capsule(a, b, d):
    return Manifold.batch_hull([sphere(d, a), sphere(d, b)])


def both_sides(m):
    return m + m.mirror((1, 0, 0))


def union(parts):
    return Manifold.batch_boolean(parts, OpType.Add) if len(parts) > 1 else parts[0]


def annular_sector(r0, r1, z0, z1, a0, deg):
    """Ring segment r0..r1, z0..z1, from angle a0 (deg, from +x) spanning deg."""
    cs = CrossSection.square((r1 - r0, z1 - z0)).translate((r0, z0))
    return Manifold.revolve(cs, 2 * SEG, deg).rotate((0, 0, a0))


# ----------------------------------------------------------------- derived ---
z_top = g.holder_len + g.rod_cover
r_bore = (g.holder_d + g.fit_clear) / 2
z_plen_top = g.land + g.conv + g.plen_h
Hn = z_plen_top + g.wall
th = math.radians(g.theta)
dirv = np.array([math.sin(th), 0.0, math.cos(th)])
aim = np.array([0.0, 0.0, -g.grid_z + g.aim_dz])
E = aim + g.standoff * dirv
I = E + Hn * dirv
B = np.array([g.ring_r, 0.0, I[2] + g.arm_rise])
zc = -g.manif_h / 2
seat_h = g.port_depth + 2.5
seat_z0 = z_top - seat_h
z_j0 = seat_z0 - (g.port_y - g.ring_r) - 1.0        # start of the 45-degree jog
z_j1 = z_j0 + (g.port_y - g.ring_r)                  # end of the jog
detent_angle = g.slot_angle + 180             # slot opening away from the port (+y)
clamp_angle = g.slot_angle + 90 if abs(((g.slot_angle + 90) % 360) - 90) > 1 else g.slot_angle - 90

if B[2] > zc - g.arm_d:
    sys.exit("grid_z too small for this nozzle geometry")
if g.holder_len <= g.collar_h + 2:
    sys.exit("holder_len must be larger than collar_h + 2")
if g.ring_r - g.ch_d / 2 - r_bore < 1.2:
    sys.exit("ring_r too small: split channel too close to the bore")


# ------------------------------------------------------------------ nozzle ---
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


def arm(d):
    return capsule(I, B, d) + capsule(B, (g.ring_r, 0, zc), d)


# -------------------------------------------------------------------- body ---
def body_outer():
    lower = cyl(g.manif_h + g.collar_h, 2 * g.body_r, -g.manif_h, seg=2 * SEG)
    upper = cyl(z_top - g.collar_h, 2 * g.upper_r, g.collar_h, seg=2 * SEG)
    # 45-degree shoulder so the part prints upside down without support
    step = g.body_r - g.upper_r
    shoulder = cone(step, 2 * g.body_r, 2 * g.upper_r, g.collar_h, seg=2 * SEG)
    # feed rib: stays inside the r = body_r envelope at the bottom and only
    # widens into the fitting seat at the top (45-degree underside, no support)
    def post(d, y, z0, z1):
        return cyl(z1 - z0, d, z0).translate((0, y, 0))
    rib = Manifold.batch_hull([
        post(g.col_d, g.ring_r, g.collar_h - 1, z_top),
        post(g.col_d, g.upper_r - 3, g.collar_h - 1, z_top),
    ])
    reach = (g.port_y + g.boss_d / 2) - (g.ring_r + g.col_d / 2)
    seat = Manifold.batch_hull([
        post(g.boss_d, g.port_y, seat_z0, z_top),
        post(g.col_d, g.ring_r, seat_z0 - reach - 1, z_top),
    ])
    column = rib + seat
    return union([lower, upper, shoulder, column])


def body_inner():
    d_hb = 2 * r_bore
    d_rb = g.rod_d + g.rod_clear
    c = g.chamfer
    parts = [
        # through-bore over the holder, with lead-in chamfer at the bottom
        cyl(g.holder_len + g.manif_h + 1, d_hb, -g.manif_h - 1, seg=2 * SEG),
        cone(c, d_hb + 2 * c, d_hb, -g.manif_h, seg=2 * SEG),
        # rod bore (the step at z = holder_len sits on the holder's top face)
        cyl(z_top - g.holder_len + 1, d_rb, g.holder_len - 0.01, seg=2 * SEG),
        cone(c, d_rb, d_rb + 2 * c, z_top - c, seg=2 * SEG),
        # U-shaped symmetric split from +y (port) to the arms at +x and -x
        # teardrop section, point toward -z: prints without overhang upside down
        Manifold.revolve((CrossSection.circle(g.ch_d / 2, SEG // 2)
                          + CrossSection.square((0.05, 0.05), center=True)
                          .translate((0, -g.ch_d / 2 * math.sqrt(2)))).hull()
                         .translate((g.ring_r, 0)), 2 * SEG, 180).translate((0, 0, zc)),
        sphere(g.ch_d, (g.ring_r, 0, zc)), sphere(g.ch_d, (-g.ring_r, 0, zc)),
        # M5 port at the top -> 45-degree jog -> straight down the rib into the
        # middle of the U-split (no radial link, nothing sticks out at the bottom)
        cyl(g.port_depth + 1, g.port_pilot, z_top - g.port_depth).translate((0, g.port_y, 0)),
        cone((g.port_pilot - g.ch_d) / 2 + 0.01, g.ch_d, g.port_pilot,              # drill-point
             z_top - g.port_depth - (g.port_pilot - g.ch_d) / 2).translate((0, g.port_y, 0)),
        cyl(z_top - g.port_depth - z_j1 + 0.5, g.ch_d, z_j1).translate((0, g.port_y, 0)),
        capsule((0, g.ring_r, z_j0), (0, g.port_y, z_j1), g.ch_d),
        cyl(z_j0 - zc, g.ch_d, zc).translate((0, g.ring_r, 0)),
        sphere(g.ch_d, (0, g.ring_r, zc)),
    ]
    if g.detent == 2:                                       # tapped M4 hole
        parts.append(cyl(g.body_r + 5, g.plunger_pilot).rotate((0, 90, 0))
                     .translate((0, 0, g.detent_z)).rotate((0, 0, detent_angle)))
    if g.clamp:
        parts.append(cyl(g.body_r + 3, g.clamp_pilot).rotate((0, 90, 0))
                     .translate((0, 0, g.collar_h / 2)).rotate((0, 0, clamp_angle)))
    return union(parts)


# ------------------------------------------------------------------ detent ---
# A thin spring band, part of the bore wall, anchored at BOTH ends and free in
# between, carries the rounded bump at its middle. Anchored at both ends, its
# first printed layer is a bridge rather than a floating cantilever. The band
# runs around the bore, so it bends along the printed layers. The pocket behind
# it has a 45-degree roof (in the upside-down print orientation) so it needs no
# support either. Built with the bump at angle 0, then rotated to the slot.
half = g.tab_arc / 2
gap_deg = math.degrees(g.tab_gap / r_bore)
z0, z1 = g.detent_z - g.tab_h / 2, g.detent_z + g.tab_h / 2


def detent_pocket():
    r_out = r_bore + g.tab_t + g.tab_gap
    z_lo = z0 - g.tab_gap
    depth = r_out - (r_bore - 1)                      # 45-degree roof
    cs = CrossSection([[(r_bore - 1, z_lo - depth), (r_out, z_lo),
                        (r_out, z1 + g.tab_gap), (r_bore - 1, z1 + g.tab_gap)]])
    return (Manifold.revolve(cs, 2 * SEG, g.tab_arc)
            .rotate((0, 0, -half)).rotate((0, 0, detent_angle)))


def detent_arm():
    beam = annular_sector(r_bore, r_bore + g.tab_t, z0, z1, -half - 3, g.tab_arc + 6)
    bump = (Manifold.cylinder(g.bump_w, g.bump_r, circular_segments=SEG, center=True)
            .rotate((90, 0, 0))
            .translate((r_bore - g.bump_h + g.bump_r, 0, g.detent_z)))
    bump = bump ^ cyl(g.tab_h, 2 * (r_bore + g.tab_t), z0, seg=2 * SEG)
    return (beam + bump).rotate((0, 0, detent_angle))


# ---------------------------------------------------------------- assemble ---
outer = body_outer() + both_sides(place_nozzle(nozzle_outer()) + arm(g.arm_d))
inner = body_inner() + both_sides(place_nozzle(nozzle_inner()) + arm(g.ch_d))
if g.detent == 1:
    inner = inner + detent_pocket()
part = outer - inner
if g.detent == 1:
    part = part + detent_arm()

if part.status().name != "NoError" or part.is_empty():
    sys.exit(f"Geometry error: {part.status()}")


# ------------------------------------------------------------------ checks ---
def tweezer_model():
    slope = (g.tw_t10 - g.tw_t5) / 5.0
    t0 = g.tw_t5 - 5 * slope
    z_tip = -g.grid_z + g.tip_above_center
    L = 35.0
    return Manifold.batch_hull([
        cube((max(t0, 0.2), g.tw_w, 0.01), (0, 0, z_tip)),
        cube((t0 + slope * L, g.tw_w, 0.01), (0, 0, z_tip + L)),
    ])


bb = part.bounding_box()
_low = part ^ cyl(g.manif_h + g.collar_h, 80, -g.manif_h, seg=8)
_lb = _low.bounding_box()
max_r_low = max(abs(_lb[0]), abs(_lb[1]), _lb[3], _lb[4])
ethane_surface = -g.grid_z + g.plunge_depth
gap_tw = part.min_gap(tweezer_model(), 20.0)
lip_a = g.slot_g / 2 + g.lip
lip_gap = 2 * (E[0] - lip_a * math.cos(th))
contact = math.sqrt(max(g.bump_r ** 2 - (g.bump_r - g.bump_h) ** 2, 0))
L_arm = math.radians(g.tab_arc) * (r_bore + g.tab_t / 2)
k_arm = 192 * 3500 * (g.tab_h * g.tab_t ** 3 / 12) / L_arm ** 3    # fixed-fixed, PLA E ~ 3.5 GPa

print(f"holder_len = {g.holder_len} mm (sets nozzle height)")
print(f"Part length:                 {bb[5] - bb[2]:.1f} mm (top {z_top:.1f} mm above holder bottom)")
print(f"Lowest point of part:        {bb[2] + g.grid_z:.2f} mm above grid centre")
print(f"Clearance to ethane surface: {bb[2] - ethane_surface:.2f} mm")
print(f"Min gap to tweezer model:    {gap_tw:.2f} mm (lower 35 mm, linear taper)")
print(f"Free gap between nozzle lips:{lip_gap:.2f} mm (what must pass between them)")
print(f"Gap arms <-> lock ring:      {g.ring_r - g.arm_d / 2 - g.lock_ring_t / 2:.2f} mm per side")
if g.detent == 2:
    rb2 = g.plunger_ball / 2
    c2 = math.sqrt(max(rb2 ** 2 - (rb2 - g.plunger_proud) ** 2, 0))
    print(f"Detent: M4 ball spring plunger at {detent_angle % 360:.0f} deg, {g.detent_z} mm above "
          f"holder bottom; releases after ~{g.slot_len - g.detent_z - c2:.1f} mm of lift")
    print(f"Plunger hole: M4 tap, {g.body_r - r_bore:.1f} mm of thread; a 9.8 mm plunger "
          f"set flush with the bore sticks out ~{9.8 - (g.body_r - r_bore):.1f} mm")
if g.detent == 1:
    print(f"Detent: clicks into the slot opening at {detent_angle % 360:.0f} deg; releases after "
          f"~{g.slot_len - g.detent_z - contact:.1f} mm of lift")
    print(f"Detent spring: ~{k_arm:.1f} N/mm, ~{k_arm * g.bump_h:.1f} N radial at full bump "
          f"(PLA estimate)")
print(f"Max radius below the shoulder: {max_r_low:.1f} mm (lower body {g.body_r} mm)")
print(f"Lift to clear the holder's bottom end: {g.manif_h + g.chamfer:.1f} mm")
print(f"Part volume:                 {part.volume() / 1000:.1f} cm^3 (~{part.volume() / 1000 * 1.24:.0f} g PLA)")
for label, val, lim in [("ethane", bb[2] - ethane_surface, 3.0), ("tweezer", gap_tw, 1.0)]:
    if val < lim:
        print(f"WARNING: {label} clearance below {lim} mm")


# ------------------------------------------------------------------- write ---
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
        f.write(b"airknife manifold v2".ljust(80, b" "))
        f.write(np.uint32(len(t)).tobytes())
        f.write(rec.tobytes())
    print(f"Wrote {path} ({len(t)} triangles)")


write_stl(part, g.out)
