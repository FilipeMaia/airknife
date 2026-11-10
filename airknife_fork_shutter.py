#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "manifold3d>=3.0",
#     "numpy>=1.24",
# ]
# ///
"""
One-piece bracket: the downward-blowing nozzle fork (airknife_breadboard.py)
plus the rails for the solenoid-pulled air shutter (airknife_shutter.py), so the
shutter can sit right under the grid.

  * The fork part is taken unchanged from airknife_breadboard.py (same nozzles,
    gas path, bolt holes and fitting). That file must sit next to this one.
  * The shutter rails hang from the same plate, `shutter_z` below the grid
    (default 10 mm, the shutter's top is then ~8 mm under the grid's lower edge).
  * The string leaves the shutter through a PTFE-tube socket in the back
    crossbar. The tube runs free in a gentle S-bend up to a hole in the plate,
    straight through a breadboard hole (on the hole row level with the grid)
    and on to the solenoid.
  * Same timing as before: the solenoid releases the plunger after 20 mm of its
    23 mm stroke; by then the shutter's front edge is well behind the tweezer.

Printing: upside down on the flat top of the fork and plate (as the fork alone).
The shutter rails then hang below the fork, so this one DOES need supports:
tree supports, "on build plate only", which reach only the rails and crossbar.

Parts:  fork_shutter.stl (print orientation), shutter.stl (prints flat, hooks
up), fork_shutter_assembly.stl (in place, for checking).

Usage:  uv run airknife_fork_shutter.py
        uv run airknife_fork_shutter.py shutter_z=-12 ...
Coordinates (mm): origin = grid centre, plunger UP; +z up; x = grid normal;
y = along the grid face, away from the breadboard.
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
    shutter_z=-10.0,      # shutter mid-plane (below the grid centre)
    front=10.0, back=-12.0, half_x=27.0, sh_t=1.6,
    travel=26.0,          # 23 mm stroke + 3 mm spare
    sol_stroke=23.0, pre_release=20.0,
    rail_w=6.0, rail_h=8.0, slot_h=2.4, slot_d=3.0,
    cross_t=6.0, tube_d=4.3, plate_tube_d=5.0,
    string_x=7.5,         # breadboard column used for the string/tube
    string_z=0.0,         # breadboard row used for the string/tube
    light_r=5.0,          # light beam radius along x through the grid (check)
    # plunger (checks)
    stroke=67.0, tw_w=9.0, tw_t=4.0, ring_top=62.5, ring_h=13.5, ring_w=13.0, ring_t=8.4,
    holder_z0=101.5, holder_d=17.0, holder_len=45.5,
    fork_script="airknife_breadboard.py", fork_args="slot_g=0.6 lip=1.0",
    segments=48, outdir=".",
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


# ------------------------------------------------------------ the fork -----
here = os.path.dirname(os.path.abspath(__file__))
fork_path = g.fork_script if os.path.isabs(g.fork_script) else os.path.join(here, g.fork_script)
if not os.path.exists(fork_path):
    sys.exit(f"Needs {g.fork_script} next to this script")
saved, sys.argv = sys.argv, [fork_path, "out=" + os.devnull] + g.fork_args.split()
with contextlib.redirect_stdout(io.StringIO()):
    F = runpy.run_path(fork_path)
sys.argv = saved
fork = F["part"]
y_back, y_front = F["y_back"], F["y_front"]
fork_half_w, fork_z0, z_top = F["half_w"], F["plate_z0"], F["z_top"]

# ------------------------------------------------------- shutter rails -----
zc = g.shutter_z
rail_in = g.half_x - g.slot_d + 0.4
rail_out = rail_in + g.rail_w
z0, z1 = zc - g.rail_h / 2, zc + g.rail_h / 2
cross_y1 = g.back - g.travel - 2.0
cross_y0 = cross_y1 - g.cross_t
if z1 > -1.525 - 3:
    sys.exit("shutter_z too high: rails would come within 3 mm of the grid")


def frame_part():
    # plate extension below the fork plate, same width and thickness
    ext = box(-fork_half_w, fork_half_w, y_back, y_front, min(z0 - 12, fork_z0), fork_z0 + 0.5)
    rails, slots, gus, posts = [], [], [], []
    for s in (1, -1):
        xa, xb = sorted((s * rail_in, s * rail_out))
        rails.append(box(xa, xb, y_front - 0.5, g.front, z0, z1))
        ca, cb = sorted((s * (rail_in - 0.01), s * (g.half_x + 0.4)))
        slots.append(box(ca, cb, cross_y1 - 0.01, g.front - 1.5, zc - g.slot_h / 2, zc + g.slot_h / 2))
        xm = s * (rail_in + g.rail_w / 2)
        gus.append(Manifold.batch_hull([box(xm - 1.5, xm + 1.5, y_front - 0.5, y_front, z0 - 10, z0 + 0.1),
                                        box(xm - 1.5, xm + 1.5, y_front - 0.5, y_front + 10, z0, z0 + 0.1)]))
        posts.append(zcyl(z0 - 5, z0 + 0.01, xm, g.front - 4, 3.0) + zcyl(z0 - 5, z0 - 4, xm, g.front - 4, 4.5))
    cross = box(-rail_out, rail_out, cross_y0, cross_y1, z0, z1)
    body = union([ext, cross] + rails + gus + posts)
    sock = ycyl(cross_y0 - 0.01, cross_y1 - 1.5, g.string_x, zc, g.tube_d)     # tube socket
    exit_ = ycyl(cross_y1 - 1.6, cross_y1 + 0.02, g.string_x, zc, 2.0)        # string exit
    ptube = ycyl(y_back - 1, y_front + 1, g.string_x, g.string_z, g.plate_tube_d)  # through the plate
    return body - union([union(slots), sock, exit_, ptube])


def shutter_part(pulled=0.0):
    sh = box(-g.half_x, g.half_x, g.back, g.front, zc - g.sh_t / 2, zc + g.sh_t / 2)
    tab = box(g.string_x - 3, g.string_x + 3, g.back - 4, g.back + 0.1, zc - g.sh_t / 2, zc + g.sh_t / 2)
    zb = zc - g.sh_t / 2
    hooks = union([zcyl(zb - 3.5, zb + 0.01, s * (g.half_x - 6), g.back + 3, 3.0)
                   + zcyl(zb - 3.5, zb - 2.5, s * (g.half_x - 6), g.back + 3, 4.5) for s in (1, -1)])
    hole = zcyl(zc - 2, zc + 2, g.string_x, g.back - 2, 1.5)
    m = (sh + tab + hooks) - hole
    return m.translate((0, -pulled, 0)) if pulled else m


frame = frame_part()
# remove the plate-tube hole from the fork plate too
part = (fork + frame) - ycyl(y_back - 1, y_front + 1, g.string_x, g.string_z, g.plate_tube_d)
sh_closed, sh_rel, sh_end = shutter_part(), shutter_part(g.pre_release), shutter_part(g.sol_stroke)


# -------------------------------------------------------------- checks -----
def plunger_parts():
    tw = box(-g.tw_t / 2, g.tw_t / 2, -g.tw_w / 2, g.tw_w / 2, 0.8, g.holder_z0)
    ring = box(-g.ring_t / 2, g.ring_t / 2, -g.ring_w / 2, g.ring_w / 2, g.ring_top - g.ring_h, g.ring_top)
    hold = zcyl(g.holder_z0, g.holder_z0 + g.holder_len, 0, 0, g.holder_d)
    grid = xcyl(-0.05, 0.05, 0, 0, 3.05)
    return {"grid+tweezer": grid + tw, "lock ring": ring, "holder": hold}


def swept(m):
    return Manifold.batch_hull([m, m.translate((0, 0, -g.stroke))])


pp = plunger_parts()
front_rel = g.front - g.pre_release
print(f"Shutter top {-(zc + g.sh_t / 2) - 1.525:.1f} mm below the grid's lower edge "
      f"(mid-plane {-zc:.0f} mm below the grid centre)")
print(f"At release ({g.pre_release:.0f} mm of stroke) the shutter's front edge is "
      f"{-g.tw_w / 2 - front_rel:.1f} mm behind the tweezer; at the end of the stroke "
      f"{-g.tw_w / 2 - (g.front - g.sol_stroke):.1f} mm")
print(f"Shutter closed vs grid/tweezer (plunger up): {sh_closed.min_gap(pp['grid+tweezer'], 30):.1f} mm")
for nm, m in [("bracket", part), ("shutter at release", sh_rel), ("shutter end of stroke", sh_end)]:
    for pn, pm in pp.items():
        sw = swept(pm)
        hit = (m ^ sw).volume() > 1e-6
        print(f"  plunge sweep: {nm:22s} vs {pn:12s} {m.min_gap(sw, 40):5.1f} mm" + ("  <-- HIT" if hit else ""))
beam = xcyl(-150, 150, 0, 0, 2 * g.light_r)
for nm, m in [("bracket", part), ("shutter (closed)", sh_closed)]:
    hit = (m ^ beam).volume() > 1e-6
    print(f"Light beam (radius {g.light_r} mm along x) vs {nm}: "
          + ("BLOCKED" if hit else f"clear by {m.min_gap(beam, 30):.1f} mm"))
print(f"String/PTFE tube: plate hole and breadboard hole at {-g.string_x:+.1f} mm "
      f"(right of axis = +), {g.string_z:.0f} mm; tube socket in the crossbar at {zc:.0f} mm "
      f"(the tube makes a {g.string_z - zc:.0f} mm S-bend over {cross_y0 - y_back:.0f} mm)")
print(f"Bracket volume {part.volume() / 1000:.1f} cm^3 (~{part.volume() / 1000 * 1.24:.0f} g PLA); "
      f"bolt holes as the fork alone")


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


out = {"fork_shutter.stl": on_bed(part.rotate((180, 0, 0))),
       "shutter.stl": on_bed(sh_closed.rotate((180, 0, 0))),
       "fork_shutter_assembly.stl": part + sh_closed}
for name, m in out.items():
    write_stl(m, os.path.join(g.outdir, name))
print("Wrote", ", ".join(out))
