#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "manifold3d>=3.0",
#     "numpy>=1.24",
# ]
# ///
"""
Air-blocking trapdoor under the grid, released by the falling lock ring.
For use with the downward-blowing breadboard fork (airknife_breadboard.py).

How it works (positions with the plunger UP, origin = grid centre):

  * A light DOOR, hinged on a horizontal axis 18 mm behind the plunge axis and
    20 mm below the grid, covers the area under the grid and catches the
    downward air while the nozzles blow.
  * The door is held shut by a nylon LINE tied to it close to its hinge. The
    line runs straight up (beside the fork) to a LEVER pivoted behind the
    plunge path.
  * The lever has a small FOOT that sits just under the 2 mm back ledge of the
    tweezer lock ring. A rubber band, mounted over-centre, holds the lever
    ARMED: foot pressed lightly up against the ring, line taut, door shut.
  * When the plunger drops, the ring pushes the foot down. The lever turns, the
    line goes slack at once and the door swings open under gravity. After 15 deg
    of lever travel the band passes over centre and snaps the lever fully back,
    clear of the tweezer and holder.
  * Reset: load the next tweezer, then push the lever back up until it clicks
    over centre. The foot rises from below to the ring and the line pulls the
    door shut by itself.

Parts (each exported in its print orientation):
  trap_frame.stl    door frame, bolts to the breadboard below the fork
  trap_door.stl     the door
  trap_trigger.stl  lever bracket, bolts to the breadboard above the fork
  trap_lever.stl    the lever
  trap_assembly.stl everything in place, armed, with the fork (for checking)

Hardware: 4x M6 bolts + large washers, 2x M2x10 screws (door hinge pins),
1x M2x16 screw (lever pivot), ~15 cm of 0.25-0.3 mm nylon fishing line, one
small rubber band (or orthodontic elastic).

Usage:  uv run airknife_trapdoor.py
        uv run airknife_trapdoor.py door_z=-22 outdir=stl ...
Coordinates (mm): origin = grid centre, plunger UP; +z up; x = grid normal;
y = along the grid face, away from the breadboard (board at y = -bb_dist).
"""
import contextlib
import io
import math
import os
import runpy
import sys

import numpy as np
from manifold3d import CrossSection, Manifold, OpType

# --------------------------------------------------------------- parameters --
P = dict(
    # breadboard (same as the fork)
    bb_dist=86.5, shim=1.0, plate_t=6.0, pitch=25.0, hole_u=-7.5, hole_z=0.0,
    bolt_d=6.6, adj=3.0, washer_r=9.0, plate_margin=11.0,
    # plunger and tweezer (measured)
    stroke=67.0, ring_top=62.5, ring_h=13.5, ring_w=13.0, ring_t=8.4,
    tw_w=9.0, tw_t=4.0, holder_z0=101.5, holder_d=17.0, holder_len=45.5,
    lnf_top=-53.0,        # floating ring top at full LN2
    # door
    door_y=-18.0,         # hinge axis position (behind the plunge axis)
    door_z=-20.0,         # hinge axis height (below the grid)
    door_tip=10.0,        # door reaches this far in front of the plunge axis
    door_half_x=24.0,     # half width along the grid normal (catches both jets)
    door_t=1.2,           # door thickness
    line_x=-21.0,         # the line runs straight up here, outside the fork's tines
    line_y=-14.0,         # ... and here (it holds the door 4 mm in front of the hinge)
    rail_w=3.0,
    # lever (pivot axis along x)
    piv_y=-25.5, piv_z=49.5,
    foot_w=6.0,           # foot width along x
    foot_depth=1.6,       # how far the foot reaches under the 2 mm ledge
    b_r=7.0, b_ang=15.0,  # band peg: radius and angle behind vertical (= toggle point)
    q_drop=14.0,          # band anchor peg below the pivot
    retract=50.0,         # lever stroke (deg) to the retract stop
    overtravel=1.0,       # foot may rise this much above the ring bottom (no tweezer)
    # output
    fork_script="airknife_breadboard.py",
    segments=48,
    outdir=".",
)

for arg in sys.argv[1:]:
    k, v = arg.split("=", 1)
    if k not in P:
        sys.exit(f"Unknown parameter: {k}")
    P[k] = v if isinstance(P[k], str) else type(P[k])(float(v))
g = type("Params", (), P)
SEG = int(g.segments)


# ------------------------------------------------------------------ helpers --
def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def xcyl(x0, x1, y, z, d, seg=SEG):
    """Cylinder along x."""
    return (Manifold.cylinder(x1 - x0, d / 2, circular_segments=seg)
            .rotate((0, 90, 0)).translate((x0, y, z)))


def zcyl(z0, z1, x, y, d, seg=SEG):
    return Manifold.cylinder(z1 - z0, d / 2, circular_segments=seg).translate((x, y, z0))


def union(parts):
    return Manifold.batch_boolean(parts, OpType.Add) if len(parts) > 1 else parts[0]


def rot_x_about(m, y0, z0, deg):
    """Rotate a manifold about the x axis through (y0, z0)."""
    return m.translate((0, -y0, -z0)).rotate((deg, 0, 0)).translate((0, y0, z0))


def rot_pt(p, y0, z0, deg):
    a = math.radians(deg)
    y, z = p[1] - y0, p[2] - z0
    return np.array([p[0], y0 + y * math.cos(a) - z * math.sin(a),
                     z0 + y * math.sin(a) + z * math.cos(a)])


def board_plate(x_c, z_c, holes, z_lo=None, z_hi=None):
    """Plate on the breadboard with oversized M6 holes; returns (solid, holes)."""
    y_back = -g.bb_dist + g.shim
    y_front = y_back + g.plate_t
    hw = max(abs(x - x_c) for x, _ in holes) + g.plate_margin
    z0 = min(min(z for _, z in holes) - g.plate_margin, z_lo if z_lo is not None else 1e9)
    z1 = max(max(z for _, z in holes) + g.plate_margin, z_hi if z_hi is not None else -1e9)
    plate = box(x_c - hw, x_c + hw, y_back, y_front, z0, z1)
    rr = CrossSection.square((2 * g.adj, 2 * g.adj), center=True).offset(
        g.bolt_d / 2, circular_segments=SEG)
    hole = Manifold.extrude(rr, g.plate_t + 2).rotate((-90, 0, 0))
    cut = union([hole.translate((x, y_back - 1, z)) for x, z in holes])
    return plate - cut, y_front


def grid_holes(x_avoid, w_avoid, z_row, extra=()):
    """Two board holes in row z_row, either side of the members (x ranges) that
    stand off the plate, so washers and the hex key stay clear of them."""
    cols = [-(g.hole_u + k * g.pitch) for k in range(-4, 5)]        # x = -u
    members = [(x_avoid - w_avoid / 2, x_avoid + w_avoid / 2)] + list(extra)
    c_ok = [c for c in cols if all(c <= a - g.washer_r - 1 or c >= b + g.washer_r + 1
                                   for a, b in members)]
    lo, hi = min(a for a, _ in members), max(b for _, b in members)
    left = max(c for c in c_ok if c <= lo)
    right = min(c for c in c_ok if c >= hi)
    rows = [g.hole_z + m * g.pitch for m in range(-6, 7)]
    z = min(rows, key=lambda r: abs(r - z_row))
    return [(left, z), (right, z)]


y_front = -g.bb_dist + g.shim + g.plate_t

# ------------------------------------------------------------- door frame ---
H = np.array([0.0, g.door_y, g.door_z])           # hinge axis point
leaf_len = g.door_tip - g.door_y
rail_in = g.door_half_x + 1.0
rail_out = rail_in + g.rail_w
bar_z0, bar_z1 = g.door_z - 4, g.door_z + 4
cross_y1 = g.door_y - 6.0                         # crossbar behind the hinge
cross_y0 = cross_y1 - 3.0
knuck_d = 5.0
leaf_z0 = g.door_z - knuck_d / 2                  # leaf flush with the knuckle bottoms
leaf_top = leaf_z0 + g.door_t                     # (door prints flat)


def frame_part():
    holes = grid_holes(0.0, 8.0, g.door_z - 5)
    plate, _ = board_plate(0.0, g.door_z, holes, z_lo=bar_z0 - 12, z_hi=bar_z1 + 2)
    stem = box(-4, 4, y_front - 0.5, cross_y1, bar_z0, bar_z1)
    gus = Manifold.batch_hull([box(-2, 2, y_front - 0.5, y_front, bar_z0 - 10, bar_z0 + 0.1),
                               box(-2, 2, y_front - 0.5, y_front + 10, bar_z0, bar_z0 + 0.1)])
    cross = box(-rail_out, rail_out, cross_y0, cross_y1, bar_z0, bar_z1)
    web = Manifold.batch_hull([box(-rail_out, rail_out, cross_y0, cross_y0 + 0.1, bar_z0, bar_z1),
                               box(-4, 4, cross_y0 - (rail_out - 4), cross_y0, bar_z0, bar_z1)])
    rails = union([box(s * rail_in if s > 0 else -rail_out, rail_out if s > 0 else -rail_in,
                       cross_y0, g.door_tip, bar_z0, bar_z1) for s in (1, -1)])
    # closed stops: tabs over the door's front corners
    stops = []
    for s in (1, -1):
        xi, xo = s * (g.door_half_x - 2), s * rail_in
        tab = box(min(xi, xo), max(xi, xo), g.door_tip - 8, g.door_tip - 2, leaf_top + 0.2, bar_z1)
        root = box(min(xo, xo + s * 0.1), max(xo, xo + s * 0.1), g.door_tip - 8 - 3, g.door_tip - 2,
                   leaf_top + 0.2, bar_z1)
        stops.append(Manifold.batch_hull([tab, root]))     # 45-deg underside when printed
    stops = union(stops)
    pins = union([xcyl(-rail_out - 1, -rail_in + 0.01, g.door_y, g.door_z, 1.75),
                  xcyl(rail_in - 0.01, rail_out + 1, g.door_y, g.door_z, 1.75)])
    return union([plate, stem, gus, cross, web, rails, stops]) - pins, holes


def door_part(open_deg=0.0):
    hx = g.door_half_x
    leaf = box(-hx, hx, g.door_y, g.door_tip, leaf_z0, leaf_top)
    knuck = union([xcyl(15.0, rail_in - 0.4, g.door_y, g.door_z, knuck_d),
                   xcyl(-(rail_in - 0.4), -15.0, g.door_y, g.door_z, knuck_d)])
    boss = zcyl(leaf_z0, leaf_z0 + 2.5, g.line_x, g.line_y, 4.0)   # line tie point
    holes = union([xcyl(14.0, rail_in, g.door_y, g.door_z, 2.4),
                   xcyl(-rail_in, -14.0, g.door_y, g.door_z, 2.4),
                   zcyl(leaf_z0 - 1, leaf_z0 + 4, g.line_x, g.line_y, 1.3)])
    d = (leaf + knuck + boss) - holes
    return rot_x_about(d, g.door_y, g.door_z, -open_deg) if open_deg else d


# ------------------------------------------------------------- lever + bracket
Pv = np.array([0.0, g.piv_y, g.piv_z])
ring_bot = g.ring_top - g.ring_h
ledge_back = -g.ring_w / 2                       # -6.5
foot_front = -g.tw_w / 2 - 0.3                   # stays 0.3 mm off the tweezer
foot_back = foot_front - g.foot_depth
A0 = np.array([g.line_x, g.line_y, g.piv_z])     # line hole on the lever (front of the pivot)
B0 = Pv + g.b_r * np.array([0.0, -math.sin(math.radians(g.b_ang)), math.cos(math.radians(g.b_ang))])
Q0 = Pv + np.array([0.0, 0.0, -g.q_drop])
LX0, LX1 = -9.0, 3.0                              # lever body x range (prints on its side)
PEG_X1 = 8.0                                      # band pegs: lever peg x 3..8, wall at 8.5+
WALL_X0, WALL_X1 = 8.5, 14.0


def lever_part(gamma=0.0):
    """gamma = retract angle (deg); 0 = armed."""
    zb = g.piv_z - 4.0                               # flat underside: prints upright
    hub = xcyl(LX0, LX1, g.piv_y, g.piv_z, 8.0) + box(LX0, LX1, g.piv_y - 4, g.piv_y + 4, zb, g.piv_z)
    arm = box(LX0, LX1, g.piv_y, foot_back, zb, ring_bot)
    foot = box(LX0, LX1, foot_back, foot_front, zb, ring_bot)
    a_arm = box(g.line_x - 2.5, LX0 + 1, A0[1] - 2, A0[1] + 2, zb, g.piv_z + 2)
    horn = Manifold.batch_hull([xcyl(LX0, LX1, g.piv_y, g.piv_z, 6.0),
                                xcyl(LX0, LX1, B0[1], B0[2], 4.0)])
    peg = xcyl(LX1 - 0.01, PEG_X1, B0[1], B0[2], 2.5) + xcyl(PEG_X1 - 1, PEG_X1, B0[1], B0[2], 4.0)
    holes = union([xcyl(LX0 - 1, LX1 + 1, g.piv_y, g.piv_z, 2.4),
                   zcyl(A0[2] - 3, A0[2] + 3, g.line_x, A0[1], 1.3)])
    lv = union([hub, arm, foot, a_arm, horn, peg]) - holes
    return rot_x_about(lv, g.piv_y, g.piv_z, -gamma) if gamma else lv


def trigger_part():
    sx0, sx1 = -22.0, LX0 - 1.0                   # spine incl. pivot cheek, clear of the fork's tube
    holes = grid_holes((sx0 + sx1) / 2, sx1 - sx0, g.piv_z, extra=[(WALL_X0, WALL_X1)])
    over_deg = math.degrees(g.overtravel / math.hypot(foot_front - g.piv_y, ring_bot - g.piv_z))
    b_up = rot_pt(B0, g.piv_y, g.piv_z, over_deg)        # lever rotated foot-up by the overtravel
    z_lo = min(g.piv_z - 4.5 - 10, Q0[2] - 3)
    plate, _ = board_plate(0.5 * (sx0 + WALL_X1), g.piv_z, holes, z_lo=z_lo,
                           z_hi=max(g.piv_z + 9, b_up[2] + 4))
    spine = box(sx0, sx1, y_front - 0.5, g.piv_y + 3.5, g.piv_z - 4.5, g.piv_z + 6.5)
    gus = Manifold.batch_hull([box(sx0 + 2, sx1 - 2, y_front - 0.5, y_front, g.piv_z - 14.5, g.piv_z - 4.4),
                               box(sx0 + 2, sx1 - 2, y_front - 0.5, y_front + 10, g.piv_z - 4.5, g.piv_z - 4.4)])
    # right wall (x > lever) carrying the band anchor and the two stops
    b_rt = rot_pt(B0, g.piv_y, g.piv_z, -g.retract)
    wall_y1 = b_rt[1] + 1.25 + 3.0
    wall = box(WALL_X0, WALL_X1, y_front - 0.5, wall_y1, Q0[2] - 3, b_up[2] + 4)
    def stop(y_face, side, zc):
        # short block (2.5 mm out from the wall) that the band peg's cap runs into
        y0, y1 = (y_face - 3, y_face) if side < 0 else (y_face, y_face + 3)
        return box(PEG_X1 - 2.0, WALL_X0 + 0.01, y0, y1, zc - 2.5, zc + 2.5)
    up_stop = stop(b_up[1] - 2.0, -1, b_up[2])          # foot-up limit (no tweezer loaded)
    rt_stop = stop(b_rt[1] + 2.0, +1, b_rt[2])          # retract limit
    qpeg = xcyl(PEG_X1 - 2.5, WALL_X0 + 0.01, Q0[1], Q0[2], 2.5) + \
        xcyl(PEG_X1 - 2.5, PEG_X1 - 1.5, Q0[1], Q0[2], 4.0)
    pin = xcyl(sx1 - 4.5, sx1 + 0.1, g.piv_y, g.piv_z, 1.75)
    return union([plate, spine, gus, wall, up_stop, rt_stop, qpeg]) - pin, holes


# ---------------------------------------------------------------- kinematics -
def foot_contact_top(gamma):
    """Highest point of the foot top that is still under the ledge."""
    best = None
    for yy in np.linspace(foot_back, foot_front, 9):
        p = rot_pt(np.array([0, yy, ring_bot]), g.piv_y, g.piv_z, -gamma)
        if ledge_back <= p[1] <= -g.tw_w / 2 and (best is None or p[2] > best[2]):
            best = p
    return best


gam, contact = 0.0, []
while gam < g.retract:
    c = foot_contact_top(gam)
    if c is None:
        break
    contact.append((gam, ring_bot - c[2]))
    gam += 0.25
g_c, s_c = contact[-1]                                 # foot leaves the ledge after s_c of plunge
a_drop_c = A0[2] - rot_pt(A0, g.piv_y, g.piv_z, -g_c)[2]
line_r = g.line_y - g.door_y                           # door lever arm of the line


def leaf_top_near_path(delta):
    best = -1e9
    for yy in (-g.tw_w / 2, 0.0, g.tw_w / 2):
        r = yy - g.door_y
        if 0 <= r <= leaf_len * math.cos(math.radians(delta)) + 1e-6:
            best = max(best, g.door_z - r * math.tan(math.radians(delta)))
    return best


def simulate(spring=0.0, dt=1e-5):
    """Grid falls at 1 g; the door is released at once (the line slackens as soon
    as the lever turns) and swings on gravity (+ an optional spring, rad/s^2).
    Returns the smallest gap between grid/tweezer and door, and when the door
    reaches 75 deg."""
    G = 9810.0
    t, d, w, worst, t75 = 0.0, 0.0, 0.0, 1e9, None
    while t < 0.2:
        acc = 1.5 * G / leaf_len * math.cos(d) + spring
        w += acc * dt
        d = min(d + w * dt, math.pi / 2)
        if t75 is None and d >= math.radians(75):
            t75 = t
        z_grid = -1.5 - 0.5 * G * t * t
        if z_grid < g.door_z - leaf_len:
            break
        worst = min(worst, z_grid - leaf_top_near_path(math.degrees(d)))
        t += dt
    return worst, t75


worst_gap, t_75 = simulate()
worst_gap_s, t_75_s = simulate(spring=2000.0)
d_final = 90.0
T0 = np.array([g.line_x, g.line_y, leaf_top])
L0 = float(np.linalg.norm(A0 - T0))


# --------------------------------------------------------------- assemble ---
frame, frame_holes = frame_part()
door_armed, door_open = door_part(0.0), door_part(d_final)
lever_armed, lever_ret = lever_part(0.0), lever_part(g.retract)
trigger, trig_holes = trigger_part()

fork = None
fork_path = g.fork_script if os.path.isabs(g.fork_script) else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), g.fork_script)
if os.path.exists(fork_path):
    saved = sys.argv
    sys.argv = [fork_path, "out=" + os.devnull]
    with contextlib.redirect_stdout(io.StringIO()):
        fork = runpy.run_path(fork_path)["part"]
    sys.argv = saved


# plunger model, plunger UP
def plunger(drop=0.0):
    rt = g.ring_top - drop
    tw = box(-g.tw_t / 2, g.tw_t / 2, -g.tw_w / 2, g.tw_w / 2, 0.8 - drop, g.holder_z0 - drop)
    ring = box(-g.ring_t / 2, g.ring_t / 2, -g.ring_w / 2, g.ring_w / 2, rt - g.ring_h, rt)
    hold = zcyl(g.holder_z0 - drop, g.holder_z0 + g.holder_len - drop, 0, 0, g.holder_d)
    grid = xcyl(-0.05, 0.05, 0, -drop, 3.05)
    return {"grid+tweezer": grid + tw, "lock ring": ring, "holder": hold}


def swept(m):
    return Manifold.batch_hull([m, m.translate((0, 0, -g.stroke))])


# ------------------------------------------------------------------- report -
print(f"Door: hinge {-g.door_y:.0f} mm behind the plunge axis, {-g.door_z:.0f} mm below the grid; "
      f"{2 * g.door_half_x:.0f} x {leaf_len:.0f} mm")
print(f"Lever: pivot {-g.piv_y:.1f} mm behind the axis, {g.piv_z:.1f} mm above the grid; foot "
      f"{g.foot_depth} mm under the ring's back ledge, at the ring bottom ({ring_bot:.1f} mm)")
print(f"Trigger: the ring pushes the foot down; the line goes slack at once (the lever end drops "
      f"{a_drop_c:.1f} mm while the foot is on the ledge, the door only needs to fall "
      f"{line_r:.0f} mm at the line)")
print(f"Door opening on gravity alone: 75 deg after {1000 * t_75:.0f} ms, grid always at least "
      f"{worst_gap:.1f} mm above it" + ("  <-- TOO CLOSE" if worst_gap < 3 else "")
      + f"; with a light opening band: 75 deg after {1000 * t_75_s:.0f} ms, gap >= {worst_gap_s:.1f} mm")
print(f"Band goes over centre after {g.b_ang:.0f} deg of lever travel (foot down "
      f"{20 * math.sin(math.radians(g.b_ang)):.0f} mm); lever stops at {g.retract:.0f} deg")
print(f"Open door hangs down to {g.door_z - leaf_len:.0f} mm (floating ring top {g.lnf_top:.0f} mm)")
print(f"Line: {L0:.0f} mm, vertical, from the lever hole down to the door's tie point "
      f"(tie it taut with the lever armed and the door shut against its stops)")
print("Frame bolt holes (x right-of-axis = -x, z):", [(round(-x, 1), z) for x, z in frame_holes],
      "  trigger bolt holes:", [(round(-x, 1), z) for x, z in trig_holes])

static = [("frame", frame), ("door (closed)", door_armed), ("trigger bracket", trigger),
          ("lever (armed)", lever_armed)]
moving = [("frame", frame), ("door (open)", door_open), ("trigger bracket", trigger),
          ("lever (retracted)", lever_ret)]
if fork is not None:
    static.append(("fork", fork))
    moving.append(("fork", fork))
    for nm, m in [("frame", frame), ("trigger bracket", trigger), ("lever (armed)", lever_armed),
                  ("lever (retracted)", lever_ret), ("door (closed)", door_armed)]:
        if (m ^ fork).volume() > 1e-6:
            print(f"  WARNING: {nm} intersects the fork")
lv_gaps = [lever_part(gm).min_gap(trigger, 5.0) for gm in np.linspace(0, g.retract, 11)]
print(f"Lever vs its bracket mid-stroke: min gap {min(lv_gaps[1:-1]):.2f} mm "
      f"(armed {lv_gaps[0]:.2f}, retracted {lv_gaps[-1]:.2f}; the stops touch the band peg at the ends)")
print("Clearance to the plunger:")
up = plunger(0.0)
for nm, m in static:
    for pn, pm in up.items():
        if nm == "lever (armed)" and pn == "lock ring":
            continue                                  # the foot touches the ring on purpose
        gap = m.min_gap(pm, 30.0)
        if gap < 1.0:
            print(f"  plunger up:   {nm:18s} vs {pn:12s} {gap:5.2f} mm" + ("  <-- HIT" if gap < 0.01 else ""))
for nm, m in moving:
    for pn, pm in up.items():
        gap = m.min_gap(swept(pm), 30.0)
        flag = "  <-- HIT" if (m ^ swept(pm)).volume() > 1e-6 else ""
        print(f"  plunge sweep: {nm:18s} vs {pn:12s} {gap:5.2f} mm{flag}")


# line clearance (sampled)
def seg_pts(a, b, n=60):
    return [a + (b - a) * t for t in np.linspace(0.08, 0.92, n)]


bodies = [frame, trigger] + ([fork] if fork is not None else [])
probe = union([Manifold.sphere(0.2, 8).translate(tuple(p)) for p in seg_pts(A0, T0)])
lg = min(b.min_gap(probe, 20.0) for b in bodies)
print(f"Line clearance to the frames and fork: {lg:.1f} mm")


# ------------------------------------------------------------------- export -
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
        f.write(os.path.basename(path).encode().ljust(80, b" "))
        f.write(np.uint32(len(t)).tobytes())
        f.write(rec.tobytes())


def on_bed(m):
    b = m.bounding_box()
    return m.translate((-(b[0] + b[3]) / 2, -(b[1] + b[4]) / 2, -b[2]))


out = {
    "trap_frame.stl": on_bed(frame.rotate((90, 0, 0))),      # plate back face down
    "trap_trigger.stl": on_bed(trigger.rotate((90, 0, 0))),
    "trap_door.stl": on_bed(door_armed),                     # flat
    "trap_lever.stl": on_bed(lever_armed),                   # upright, flat underside
    "trap_assembly.stl": union([frame, door_armed, trigger, lever_armed] + ([fork] if fork else [])),
}
for name, m in out.items():
    write_stl(m, os.path.join(g.outdir, name))
print("Wrote", ", ".join(out))
