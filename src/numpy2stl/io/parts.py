"""Models as named parts: load a 3MF's objects, a pack's files or a parts file without merging.

F-STL2NUMPY step "parts" (map2stl/docs/plans/active/F-STL2NUMPY-decompose.md). A city print
model comes as parts that say what they are - a 3MF with one object per layer, a
Micropolitan pack (``Barcelona, Spain_L_Solid.stl`` + ``..._L_Water.stl``), our own City
Model (terrain, buildings, roads, waterways, ...). :func:`load_trimesh` merges them, and the
decomposition needs them apart.

- ``load_parts``       a file, folder or list of files → ``{name: trimesh.Trimesh}``
- ``part_role``        what a part's name says it is: water, roads, buildings, terrain, model
- ``write_parts_file`` / ``read_parts_file``  the compact binary parts file the map2stl
                       Extrude viewer reads (``/api/export/model-parts``)

Parts file layout: uint32 header length, the UTF-8 JSON header
``{"parts": [{"name", "vertices", "faces"}, ...]}`` padded to 4 bytes, then per part its
float32 vertices (x, y, z) and uint32 faces.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["MESH_SUFFIXES", "load_parts", "part_role", "read_parts_file", "write_parts_file"]

MESH_SUFFIXES = (".stl", ".obj", ".3mf", ".ply", ".glb", ".gltf")

#: Name words → role, first match wins. "solid" is a Micropolitan pack's whole model
#: (terrain and buildings in one solid), not a building layer.
_ROLE_WORDS = (
    ("water", ("water", "waterway", "river", "lake", "sea", "ocean")),
    ("roads", ("road", "street", "rail")),
    ("buildings", ("building", "church", "tower", "wall", "fortification", "landmark")),
    ("terrain", ("terrain", "topo", "ground", "dem", "base")),
    ("model", ("solid", "model", "city", "merged")),
)


def part_role(name: str) -> str:
    """The role a part's name gives it: water, roads, buildings, terrain, model or other."""
    low = name.lower()
    for role, words in _ROLE_WORDS:
        if any(w in low for w in words):
            return role
    return "other"


def write_parts_file(parts: dict, path) -> None:
    """Write ``{name: mesh}`` (anything with ``vertices`` and ``faces``) as a parts file;
    empty parts are left out."""
    names = [k for k, m in parts.items() if len(m.faces)]
    header = json.dumps({"parts": [{"name": k, "vertices": len(parts[k].vertices),
                                    "faces": len(parts[k].faces)} for k in names]}).encode()
    header += b" " * (-len(header) % 4)
    with open(path, "wb") as f:
        f.write(np.uint32(len(header)).tobytes())
        f.write(header)
        for k in names:
            f.write(np.ascontiguousarray(parts[k].vertices, dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(parts[k].faces, dtype=np.uint32).tobytes())


def read_parts_file(path) -> dict:
    """``{name: trimesh.Trimesh}`` from a parts file (vertices as float64)."""
    import trimesh
    raw = Path(path).read_bytes()
    n = int(np.frombuffer(raw[:4], np.uint32)[0])
    header = json.loads(raw[4:4 + n])
    off, parts = 4 + n, {}
    for p in header["parts"]:
        v = np.frombuffer(raw, np.float32, p["vertices"] * 3, off).reshape(-1, 3)
        off += v.nbytes
        f = np.frombuffer(raw, np.uint32, p["faces"] * 3, off).reshape(-1, 3)
        off += f.nbytes
        parts[p["name"]] = trimesh.Trimesh(v.astype(np.float64), f.astype(np.int64), process=False)
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} bytes after the last part")
    return parts


def _load_file(path: Path) -> dict:
    import trimesh
    if path.suffix.lower() == ".parts":
        return read_parts_file(path)
    if path.suffix.lower() == ".3mf":
        from .readers import read3MF
        objs = {k: trimesh.Trimesh(v, f, process=False) for k, (v, f) in read3MF(path).items()}
        if len(objs) == 1:
            return {path.stem: next(iter(objs.values()))}
        return {f"{path.stem}:{k}": m for k, m in objs.items()}
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        meshes = {k: g for k, g in loaded.geometry.items() if isinstance(g, trimesh.Trimesh)}
        # Objects in place, as the scene places them.
        dumped = loaded.dump(concatenate=False)
        if len(dumped) == len(meshes):
            meshes = {k: m for k, m in zip(meshes, dumped, strict=True)}
        if len(meshes) == 1:
            return {path.stem: next(iter(meshes.values()))}
        return {f"{path.stem}:{k}": m for k, m in meshes.items()}
    return {path.stem: loaded}


def load_parts(source) -> dict:
    """``{name: trimesh.Trimesh}`` for a model given as parts, without merging them.

    *source* is a mesh file (a 3MF or OBJ with several objects gives one part per object,
    named ``file:object``), a ``.parts`` file, a folder (every mesh file in it, by name),
    or a list of files. A one-object file is one part named after the file's stem.
    """
    if isinstance(source, (list, tuple)):
        paths = [Path(p) for p in source]
    else:
        src = Path(source)
        paths = (sorted(p for p in src.iterdir() if p.suffix.lower() in MESH_SUFFIXES + (".parts",))
                 if src.is_dir() else [src])
    parts: dict = {}
    for p in paths:
        for name, mesh in _load_file(p).items():
            parts[name if name not in parts else f"{name}#{len(parts)}"] = mesh
    if not parts:
        raise ValueError(f"no mesh parts in {source!r}")
    return parts
