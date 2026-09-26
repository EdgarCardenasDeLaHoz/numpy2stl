"""Fourier–Mellin prototype evaluation.

Two ways to exercise `align.fourier_mellin.fourier_mellin_register`:

1. SYNTHETIC (default, no network):  build a city-like height raster, apply a
   *known* rotation + scale, and check Fourier–Mellin recovers it.  Run on both
   a GRID layout and an IRREGULAR (random-orientation) layout — the irregular
   case is exactly where the gradient-histogram path fails, so it is the point
   of the prototype.

       python -m numpy2stl.registration.scripts.fourier_mellin_prototype

2. REAL STL/OSM (needs the registration extras + network):  compare the
   Fourier–Mellin (angle, scale) against the production gradient+sweep result.

       python -m numpy2stl.registration.scripts.fourier_mellin_prototype \\
           --stl path/to/city.stl --region "Boston, MA, USA"
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from numpy2stl.registration.align.fourier_mellin import fourier_mellin_register


def _synthetic_city(n: int = 512, *, grid: bool, seed: int = 0) -> np.ndarray:
    """Height raster of rectangular 'buildings'.

    grid=True  → all footprints axis-aligned (a street grid).
    grid=False → each footprint at a random orientation (irregular city).
    """
    import cv2
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n), dtype=np.float32)
    n_buildings = 120
    for _ in range(n_buildings):
        cx, cy = rng.uniform(0.15, 0.85, 2) * n
        bw, bh = rng.uniform(0.02, 0.06, 2) * n
        ang = 0.0 if grid else rng.uniform(0, 180)
        height = rng.uniform(5, 60)
        box = cv2.boxPoints(((cx, cy), (bw, bh), ang)).astype(np.int32)
        cv2.fillConvexPoly(img, box, float(height))
    return img


def _apply(img: np.ndarray, angle_deg: float, scale: float) -> np.ndarray:
    import cv2
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, scale)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _run_synthetic() -> int:
    cases = [
        ("rot +0  scale 1.00", 0.0, 1.00),
        ("rot +7  scale 1.00", 7.0, 1.00),
        ("rot +30 scale 1.00", 30.0, 1.00),
        ("rot +0  scale 1.30", 0.0, 1.30),
        ("rot +0  scale 0.75", 0.0, 0.75),
        ("rot +18 scale 1.25", 18.0, 1.25),
        ("rot -22 scale 0.80", -22.0, 0.80),
    ]
    worst = 0.0
    for layout in ("grid", "irregular"):
        base = _synthetic_city(grid=(layout == "grid"))
        print(f"\n=== {layout.upper()} layout "
              f"(gradient-histogram {'works' if layout == 'grid' else 'FAILS'} here) ===")
        print(f"{'case':<22} {'applied':<18} {'recovered':<22} "
              f"{'rot_err':>8} {'scale_err':>9} {'resp':>6}")
        print("-" * 92)
        for label, ang, sc in cases:
            # Source is rotated/scaled by (ang, sc); registering source→base must
            # recover the INVERSE: -ang and 1/sc.
            moved = _apply(base, ang, sc)
            r = fourier_mellin_register(moved, base)
            exp_ang = ((-ang + 180.0) % 360.0) - 180.0
            exp_sc = 1.0 / sc
            rot_err = abs(((r.angle_deg - exp_ang + 180.0) % 360.0) - 180.0)
            sc_err = abs(r.scale - exp_sc)
            worst = max(worst, rot_err)
            print(f"{label:<22} a={ang:+5.0f} s={sc:.2f}    "
                  f"a={r.angle_deg:+7.2f} s={r.scale:.3f}   "
                  f"{rot_err:>7.2f}° {sc_err:>8.3f} {r.confidence:>6.2f}")
    print(f"\nWorst rotation error across all cases/layouts: {worst:.2f}°")
    print("Grid-free success criterion: irregular-layout rot_err comparable to grid.")
    return 0


def _run_crop() -> int:
    """Random-crop recovery: transform a base image by a known (angle, scale,
    translation), then keep only a random sub-window (PARTIAL overlap) and ask
    FM to detect the transform it came from.  Success = the recovered transform
    composed with the ground-truth transform is the identity."""
    import cv2
    h = w = 512
    center = (w / 2.0, h / 2.0)

    def _gt(angle, scale, tx, ty):
        M = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float64)
        M[0, 2] += tx
        M[1, 2] += ty
        return M

    def _compose(outer, inner):
        return (np.vstack([outer, [0, 0, 1]]) @ np.vstack([inner, [0, 0, 1]]))[:2]

    def _resid_err(M_rec, M_gt):
        R = _compose(M_rec, M_gt)               # should be identity
        a, b = R[0, 0], R[0, 1]
        ang = abs(np.degrees(np.arctan2(b, a)))
        sc = abs(np.hypot(a, b) - 1.0)
        m = R @ np.array([center[0], center[1], 1.0])
        trans = float(np.hypot(m[0] - center[0], m[1] - center[1]))
        return ang, sc, trans

    # (label, angle, scale, tx, ty, crop_fraction)
    cases = [
        ("crop .6, no transform", 0.0, 1.00,   0,   0, 0.6),
        ("crop .6, rot +15",     15.0, 1.00,   0,   0, 0.6),
        ("crop .6, scale 1.2",    0.0, 1.20,   0,   0, 0.6),
        ("crop .5, rot+20 sc1.2",20.0, 1.20,  20, -10, 0.5),
        ("crop .5, rot-25 sc0.85",-25.0,0.85, -15,  25, 0.5),
        ("crop .4, rot+10 sc1.1",10.0, 1.10,  30,  10, 0.4),
    ]
    for layout in ("grid", "irregular"):
        base = _synthetic_city(grid=(layout == "grid"), seed=1)
        print(f"\n=== RANDOM-CROP recovery, {layout.upper()} layout ===")
        print(f"{'case':<24} {'overlap%':>8} {'no-refine err(°/s/px)':>24}   {'+refine err(°/s/px)':>24}")
        print("-" * 92)
        rng = np.random.default_rng(7)
        for label, ang, sc, tx, ty, frac in cases:
            M_gt = _gt(ang, sc, tx, ty)
            moved = cv2.warpAffine(base, M_gt, (w, h), flags=cv2.INTER_LINEAR)
            # Keep only a random sub-window → partial overlap, same frame.
            cw, ch = int(w * frac), int(h * frac)
            x0 = int(rng.integers(0, w - cw)); y0 = int(rng.integers(0, h - ch))
            crop = np.zeros_like(moved); crop[y0:y0 + ch, x0:x0 + cw] = moved[y0:y0 + ch, x0:x0 + cw]
            overlap_pct = 100.0 * float((crop != 0).sum()) / (h * w)

            r0 = fourier_mellin_register(crop, base, refine_overlap=False)
            r1 = fourier_mellin_register(crop, base, refine_overlap=True)
            e0 = _resid_err(r0.transform, M_gt)
            e1 = _resid_err(r1.transform, M_gt)
            print(f"{label:<24} {overlap_pct:>7.1f}% "
                  f"{e0[0]:>8.2f}/{e0[1]:.3f}/{e0[2]:>5.1f}   "
                  f"{e1[0]:>8.2f}/{e1[1]:.3f}/{e1[2]:>5.1f}")
    print("\nLower err = transform recovered.  '+refine' = overlap-masking loop;")
    print("compare the two columns to see whether overlap refinement helps.")
    return 0


def _run_real(stl: str, region: str, resolution: int, z_axis: int) -> int:
    from numpy2stl.registration import register_city_stl

    parts = region.split(",")
    city = region
    if len(parts) == 4:
        try:
            city = tuple(float(x) for x in parts)
        except ValueError:
            pass

    print("Production registration (gradient + scale sweep)…")
    rep = register_city_stl(stl_file=stl, city_name=city, resolution=resolution,
                            stl_z_axis=z_axis, out_dir=False)
    prod_scale = rep.registration.scale
    prod_rot = rep.registration.angle_deg

    print("Fourier-Mellin on the same heightmaps...")
    fm = fourier_mellin_register(rep.stl_heightmap, rep.osm_heightmap)
    fm_ov = fourier_mellin_register(rep.stl_heightmap, rep.osm_heightmap,
                                    refine_overlap=True)

    # Mitigation for cross-modality: high-pass both first so the STL (terrain +
    # full geometry) and OSM (flat footprints) present comparable structure to
    # the magnitude spectrum.
    import cv2
    def _hp(a):
        a = np.nan_to_num(a.astype(np.float32))
        sig = max(15.0, min(a.shape) / 20.0)
        return a - cv2.GaussianBlur(a, (0, 0), sig)
    fm_hp = fourier_mellin_register(_hp(rep.stl_heightmap), _hp(rep.osm_heightmap),
                                    refine_overlap=True)

    print("\n                  rotation(deg)   scale   response  overlap_corr")
    print(f"production         {prod_rot:>+9.2f}   {prod_scale:.3f}        -         -")
    print(f"fourier-mellin     {fm.angle_deg:>+9.2f}   {fm.scale:.3f}     {fm.confidence:.3f}     {fm.overlap_corr:.3f}")
    print(f"  + overlap-refine {fm_ov.angle_deg:>+9.2f}   {fm_ov.scale:.3f}     {fm_ov.confidence:.3f}     {fm_ov.overlap_corr:.3f}")
    print(f"  + hp + overlap   {fm_hp.angle_deg:>+9.2f}   {fm_hp.scale:.3f}     {fm_hp.confidence:.3f}     {fm_hp.overlap_corr:.3f}")
    best = max((fm, fm_ov, fm_hp), key=lambda r: r.overlap_corr)
    print(f"\nbest vs production:  d_rotation {abs(best.angle_deg - prod_rot):.2f} deg   "
          f"d_scale {abs(best.scale - prod_scale):.3f}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stl", help="STL/3mf path (real-data mode)")
    p.add_argument("--region", help="city name or 'N,S,E,W' bbox (real-data mode)")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--z-axis", type=int, default=2)
    p.add_argument("--crop", action="store_true",
                   help="run only the random-crop / partial-overlap recovery test")
    args = p.parse_args(argv)

    if args.stl and args.region:
        return _run_real(args.stl, args.region, args.resolution, args.z_axis)
    if args.crop:
        return _run_crop()
    _run_synthetic()
    return _run_crop()


if __name__ == "__main__":
    sys.exit(main())
