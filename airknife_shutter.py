#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "manifold3d>=3.0",
#     "numpy>=1.24",
# ]
# ///
"""
Sliding air shutter under the grid, pulled open by the plunger's release
solenoid through a string. For use with the downward-blowing breadboard fork
(airknife_breadboard.py).

How it works (positions with the plunger UP, origin = grid centre):

  * A thin SHUTTER slides in two rail channels 25 mm below the grid. Closed, it
    covers the area under the grid and catches the downward air; the air just
    presses it onto the rails, so nothing has to hold it shut against the air.
  * A string tied to its back edge runs straight back, through a PTFE tube
    inside the frame's stem and through a breadboard hole, to the solenoid
    (redirected behind the board so the solenoid's downward stroke pulls it).
  * The solenoid releases the plunger after 20 mm of its 23 mm stroke. With
    the string taut, the first 20 mm already pull the shutter clear of the
    tweezer, so the shutter is open BEFORE the grid starts to fall. The rails
    are long enough for the full stroke, so the solenoid never hits a stop.
  * Two small rubber bands pull the shutter shut again when the solenoid
    relaxes (remove the tweezer first).

Parts:  shutter_frame.stl  (prints upside down on its flat top face, no supports)
        shutter.stl        (prints flat, band hooks up)
        shutter_assembly.stl (frame + shutter closed + fork, for checking)
Hardware: 2x M6 bolts + large washers, ~4 mm OD / 2 mm ID PTFE tube (~10 cm +
whatever you need behind the board), nylon line or thin cable, 2 small rubber
bands.

Usage:  uv run airknife_shutter.py
        uv run airknife_shutter.py front=11 shutter_z=-25 ...
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

P = dict(
    # breadboard (same as the fork)
    bb_dist=86.5, shim=1.0, plate_t=6.0, pitch=25.0, hole_u=-7.5, hole_z=0.0,
    bolt_d=6.6, adj=3.0, washer_r=9.0, plate_margin=11.0,
    # plunger / tweezer / cryostat
    stroke=67.0, tw_w=9.0, tw_t=4.0, ring_top=62.5, ring_h=13.5, ring_w=13.0, ring_t=8.4,
    holder_z0=101.5, holder_d=17.0, holder_len=45.5, lnf_top=-53.0,
    # shutter
    shutter_z=-25.0,      # level with a breadboard hole row, so the string runs straight
    front=10.0,           # closed: front edge this far in front of the plunge axis
    back=-12.0,           # closed: back edge (the air sheet is ~+/-8 mm wide at this depth)
    half_x=27.0,          # half width (catches both crossing jets at this depth)
    sh_t=1.6,             # shutter thickness
    travel=26.0,          # shutter travel allowed by the rails (solenoid stroke 23 mm + margin)
    stroke_sol=23.0,      # full solenoid stroke (measured)
    pre_release=20.0,     # solenoid travel before it releases the plunger (measured)
    clear_margin=3.0,     # shutter front edge must be this far behind the tweezer at release
    # frame
    rail_w=6.0, rail_h=8.0, slot_h=2.4, slot_d=3.0,
    stem_w=10.0, tube_d=4.3,
    # output
    fork_script="airknife_breadboard.py", segments=48, outdir=".",
)
for arg in sys.argv[1:]:
    k, v = arg.split("=", 1)
    if k not in P:
        sys.exit(f"Unknown parameter: {k}")
    P[k] = v if isinstance(P[k], str) else type(P[k])(float(v))
g = type("Params", (), P)
SEG = int(g.segments)


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def zcyl(z0, z1, x, y, d):
    return Manifold.cylinder(z1 - z0, d / 2, circular_segments=SEG).translate((x, y, z0))


def ycyl(y0, y1, x, z, d):
    return (Manifold.cylinder(y1 - y0, d / 2, circular_segments=SEG)
            .rotate((-90, 0, 0)).translate((x, y0, z)))


def xcyl(x0, x1, y, z, d):
    return (Manifold.cylinder(x1 - x0, d / 2, circular_segments=SEG)
            .rotate((0, 90, 0)).translate((x0, y, z)))


def union(parts):
    return Manifold.batch_boolean(parts, OpType.Add) if len(parts) > 1 else parts[0]


# -------------------------------------------------------------- layout ------
cols = [-(g.hole_u + k * g.pitch) for k in range(-4, 5)]          # board columns, x = -u
rows = [g.hole_z + m * g.pitch for m in range(-6, 7)]
row = min(rows, key=lambda r: abs(r - g.shutter_z))
zc = row                                         # shutter mid-plane on the hole row
string_x = min(cols, key=lambda c: abs(c))       # nearest column to the plunge axis
bolt_rows = [r for r in rows if r <= zc - g.rail_h / 2 - g.washer_r - 1]   # below the rails
bolt_z = max(bolt_rows)
bolt_x = [max(c for c in cols if c <= string_x - g.stem_w / 2 - g.washer_r - 1),
          min(c for c in cols if c >= string_x + g.stem_w / 2 + g.washer_r + 1)]
y_back = -g.bb_dist + g.shim
y_front = y_back + g.plate_t
sh_len = g.front - g.back
rail_y1 = g.front
cross_y1 = g.back - g.travel - 2.0               # back crossbar: behind the fully open shutter
cross_y0 = cross_y1 - 3.0
rail_in = g.half_x - g.slot_d + 0.4              # inner rail face (shutter edge runs 3 mm deep)
rail_out = rail_in + g.rail_w
z0, z1 = zc - g.rail_h / 2, zc + g.rail_h / 2
if cross_y0 < y_front + 2:
    sys.exit("bb_dist too small for this travel")


def teardrop_y(y0, y1, x, z, d):
    """Horizontal hole along y with a 45-degree point toward -z (prints upside down)."""
    r = d / 2
    cs = (CrossSection.circle(r, SEG) + CrossSection.square((0.05, 0.05), center=True)
          .translate((0, r * math.sqrt(2)))).hull()
    return Manifold.extrude(cs, y1 - y0).rotate((-90, 0, 0)).translate((x, y0, z))


def frame_part():
    # everything shares the flat top face z1, so the frame prints upside down
    xs = [-rail_out, rail_out] + bolt_x
    hw0, hw1 = min(xs) - g.plate_margin, max(xs) + g.plate_margin
    plate = box(hw0, hw1, y_back, y_front, bolt_z - g.plate_margin, z1)
    rr = CrossSection.square((2 * g.adj, 2 * g.adj), center=True).offset(g.bolt_d / 2, circular_segments=SEG)
    rr = (rr + CrossSection.square((0.05, 0.05), center=True)
          .translate((0, g.adj + g.bolt_d / 2 * math.sqrt(2)))).hull()   # point ends up toward -z
    holes = union([Manifold.extrude(rr, g.plate_t + 2).rotate((-90, 0, 0)).translate((x, y_back - 1, bolt_z))
                   for x in bolt_x])
    sx0, sx1 = string_x - g.stem_w / 2, string_x + g.stem_w / 2
    stem = box(sx0, sx1, y_front - 0.5, cross_y0 + 0.1, z0, z1)
    cross = box(-rail_out, rail_out, cross_y0, cross_y1, z0, z1)
    rails, slots, gus, posts = [], [], [], []
    for s in (1, -1):
        xa, xb = sorted((s * rail_in, s * rail_out))
        rails.append(box(xa, xb, y_front - 0.5, rail_y1, z0, z1))
        ca, cb = sorted((s * (rail_in - 0.01), s * (g.half_x + 0.4)))
        slots.append(box(ca, cb, cross_y1 - 0.01, rail_y1 - 1.5, zc - g.slot_h / 2, zc + g.slot_h / 2))
        xm = s * (rail_in + g.rail_w / 2)
        gus.append(Manifold.batch_hull([box(xm - 1.5, xm + 1.5, y_front - 0.5, y_front, z0 - 12, z0 + 0.1),
                                        box(xm - 1.5, xm + 1.5, y_front - 0.5, y_front + 12, z0, z0 + 0.1)]))
        # rubber-band post under the rail front
        posts.append(zcyl(z0 - 5, z0 + 0.01, xm, rail_y1 - 4, 3.0) + zcyl(z0 - 5, z0 - 4, xm, rail_y1 - 4, 4.5))
    gus.append(Manifold.batch_hull([box(string_x - 1.5, string_x + 1.5, y_front - 0.5, y_front, z0 - 12, z0 + 0.1),
                                    box(string_x - 1.5, string_x + 1.5, y_front - 0.5, y_front + 12, z0, z0 + 0.1)]))
    body = union([plate, stem, cross] + rails + gus + posts)
    tube = teardrop_y(y_back - 1, cross_y0 + 1.0, string_x, zc, g.tube_d)     # PTFE tube seat
    exit_ = teardrop_y(cross_y0 + 0.9, cross_y1 + 0.02, string_x, zc, 2.0)    # string exit
    return body - union([holes, union(slots), tube, exit_])


def shutter_part(pulled=0.0):
    sh = box(-g.half_x, g.half_x, g.back, g.front, zc - g.sh_t / 2, zc + g.sh_t / 2)
    tab = box(string_x - 3, string_x + 3, g.back - 4, g.back + 0.1, zc - g.sh_t / 2, zc + g.sh_t / 2)
    zb = zc - g.sh_t / 2
    hooks = union([zcyl(zb - 3.5, zb + 0.01, s * (g.half_x - 6), g.back + 3, 3.0)
                   + zcyl(zb - 3.5, zb - 2.5, s * (g.half_x - 6), g.back + 3, 4.5) for s in (1, -1)])
    hole = zcyl(zc - 2, zc + 2, string_x, g.back - 2, 1.5)
    m = (sh + tab + hooks) - hole
    return m.translate((0, -pulled, 0)) if pulled else m


frame = frame_part()
sh_closed = shutter_part(0.0)
sh_release = shutter_part(g.pre_release)
if g.travel < g.stroke_sol + 1:
    sys.exit("travel must exceed the solenoid stroke, or the solenoid will hit the back stop")
sh_open = shutter_part(g.stroke_sol)

fork = None
fork_path = g.fork_script if os.path.isabs(g.fork_script) else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), g.fork_script)
if os.path.exists(fork_path):
    saved, sys.argv = sys.argv, [fork_path, "out=" + os.devnull]
    with contextlib.redirect_stdout(io.StringIO()):
        fork = runpy.run_path(fork_path)["part"]
    sys.argv = saved


def plunger_parts():
    tw = box(-g.tw_t / 2, g.tw_t / 2, -g.tw_w / 2, g.tw_w / 2, 0.8, g.holder_z0)
    ring = box(-g.ring_t / 2, g.ring_t / 2, -g.ring_w / 2, g.ring_w / 2, g.ring_top - g.ring_h, g.ring_top)
    hold = zcyl(g.holder_z0, g.holder_z0 + g.holder_len, 0, 0, g.holder_d)
    grid = xcyl(-0.05, 0.05, 0, 0, 3.05)
    return {"grid+tweezer": grid + tw, "lock ring": ring, "holder": hold}


def swept(m):
    return Manifold.batch_hull([m, m.translate((0, 0, -g.stroke))])


# ------------------------------------------------------------------ report --
front_at_release = g.front - g.pre_release
margin = -g.tw_w / 2 - front_at_release
print(f"Shutter: {2 * g.half_x:.0f} x {sh_len:.0f} mm, {g.sh_t} mm thick, {-zc:.0f} mm below the grid "
      f"(on the breadboard hole row); rails allow {g.travel:.0f} mm of travel")
print(f"At plunger release ({g.pre_release:.0f} mm of solenoid stroke) the shutter's front edge is at "
      f"y = {front_at_release:.1f} mm, {margin:.1f} mm behind the tweezer"
      + ("" if margin >= g.clear_margin else "  <-- NOT CLEAR YET: take up string slack or shorten the shutter"))
print(f"At the end of the {g.stroke_sol:.0f} mm stroke the front edge is at y = {g.front - g.stroke_sol:.1f} mm; "
      f"the rails allow {g.travel:.0f} mm, so the solenoid never hits a stop")
print(f"String/PTFE tube: board hole at {-string_x:+.1f} mm (right of axis = +), {row:.0f} mm; "
      f"bolts at {-bolt_x[0]:+.1f} and {-bolt_x[1]:+.1f} mm, {bolt_z:.0f} mm")
print("Clearance to the plunger:")
pp = plunger_parts()
for nm, m in [("frame", frame), ("shutter (closed)", sh_closed)]:
    for pn, pm in pp.items():
        print(f"  plunger up:          {nm:18s} vs {pn:12s} {m.min_gap(pm, 40):5.1f} mm")
for nm, m in [("frame", frame), ("shutter at release", sh_release), ("shutter, full stroke", sh_open)]:
    for pn, pm in pp.items():
        sw = swept(pm)
        hit = (m ^ sw).volume() > 1e-6
        print(f"  during the plunge:   {nm:18s} vs {pn:12s} {m.min_gap(sw, 40):5.1f} mm"
              + ("  <-- HIT" if hit else ""))
if fork is not None:
    for nm, m in [("frame", frame), ("shutter", sh_closed)]:
        print(f"  fork vs {nm}: {m.min_gap(fork, 40):.1f} mm")
print(f"Shutter slides with {g.slot_h - g.sh_t:.1f} mm vertical and 0.4 mm side play in each channel")


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


out = {"shutter_frame.stl": on_bed(frame.rotate((180, 0, 0))),     # upside down, flat top on the bed
       "shutter.stl": on_bed(sh_closed.rotate((180, 0, 0))),         # hooks up
       "shutter_assembly.stl": union([frame, sh_closed] + ([fork] if fork else []))}
for name, m in out.items():
    write_stl(m, os.path.join(g.outdir, name))
print("Wrote", ", ".join(out))
