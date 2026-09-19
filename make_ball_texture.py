"""Generate an equirectangular basketball texture: pebbled orange with the classic 8-panel seam layout.

Seam layout of a regulation ball (axis through the valve = z):
  * two perpendicular great circles: the "equator" (z = 0) and one meridian (y = 0),
  * two curved seams, one in each hemisphere x>0 / x<0, that arc from the equator up around the
    pole and back: the intersection of the sphere with a circular cylinder of radius 0.5 R whose axis
    is parallel to y and passes through (x = +-0.5 R, z = 0).  Each looks like a "U" from the side and
    a circle from the pole, which is the familiar basketball panel shape.
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 2048, 1024
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "assets", "assets", "basketball.png")

u = (np.arange(W) + 0.5) / W
v = (np.arange(H) + 0.5) / H
lon = (u[None, :] * 2 - 1) * np.pi
lat = (0.5 - v[:, None]) * np.pi
x = np.cos(lat) * np.cos(lon)
y = np.cos(lat) * np.sin(lon)
z = np.sin(lat)
x, y, z = np.broadcast_arrays(x, y, z)

w = 0.026  # seam half-width (fraction of radius)
# 8-panel construction: great circle A (equator, z = 0), great circle B (meridian, y = 0), and one
# closed oval loop on each side of B, centred on B's poles (0, +-1, 0); each loop crosses A twice and
# never crosses B.  Loop = points at angular offset (a_z along the pole direction, a_x along the
# equator) from its centre satisfying (a_z/52deg)^2 + (a_x/36deg)^2 = 1.
seam = np.abs(z) < w  # A
seam |= np.abs(y) < w  # B
for cy in (1.0, -1.0):
    # angular coordinates around the loop centre c = (0, cy, 0): a_x = atan2(x, cy*y), a_z = atan2(z, cy*y)
    ax = np.arctan2(x, cy * y)
    az = np.arctan2(z, cy * y)
    f = (az / np.radians(52.0)) ** 2 + (ax / np.radians(36.0)) ** 2
    # signed distance to the loop, approximated in angle then scaled to chord length
    dist = np.abs(np.sqrt(np.maximum(f, 1e-9)) - 1.0) * np.radians(44.0)
    seam |= (dist < w) & (cy * y > 0.05)

rng = np.random.default_rng(0)
pebble = rng.uniform(-14, 14, size=(H, W))
base = np.stack([224.0 + pebble, 110.0 + pebble * 0.6, 30.0 + pebble * 0.3], -1)
black = np.array([25.0, 20.0, 18.0])
img = np.where(seam[..., None], black, base).clip(0, 255).astype(np.uint8)
os.makedirs(os.path.dirname(out), exist_ok=True)
Image.fromarray(img).save(out)
print("wrote", out)
