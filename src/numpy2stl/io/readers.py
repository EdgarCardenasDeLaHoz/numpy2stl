"""STL/OBJ/3MF mesh loading utilities — used by the stl2numpy module."""

try:
    import trimesh

    HAS_TRIMESH = True
except ImportError:
    trimesh = None
    HAS_TRIMESH = False

import numpy as np

__all__ = ["load_mesh", "load_trimesh", "read3MF"]


def load_trimesh(file_path: str):
    """Load an STL/OBJ/3MF/PLY file as one ``trimesh.Trimesh`` (scenes are concatenated;
    3MF through :func:`read3MF`, without lxml)."""
    if not HAS_TRIMESH:
        raise ImportError(
            "trimesh is required. Install with: pip install trimesh"
        )
    # Pre-warm format-specific loaders so trimesh's lazy-import cache
    # doesn't surface a stale ImportError from a prior failed attempt.
    # NB: use importlib so we don't rebind the module-level `trimesh` name
    # into a function local (which would shadow it below).
    ext = str(file_path).rsplit(".", 1)[-1].lower()
    if ext == "3mf":
        # Our own reader (standard library): trimesh's 3MF loader needs lxml, which the
        # venv does not have, and a 3MF plate (Philadelphia's) then could not be gridded.
        objs = read3MF(file_path)
        if not objs:
            raise ValueError(f"{file_path!r}: no mesh objects in the 3MF")
        return trimesh.util.concatenate([trimesh.Trimesh(v, f, process=False)
                                         for v, f in objs.values()])
    mesh = trimesh.load(file_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"Could not load {file_path!r} as a single mesh. "
            "The file may contain multiple objects; export as a merged STL."
        )
    return mesh


def load_mesh(file_path: str):
    """
    Load a mesh file and return (vertices, faces) arrays.

    Parameters
    ----------
    file_path : str
        Path to an STL, OBJ, or 3MF file.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
    faces    : ndarray of int, shape (M, 3)
    """
    mesh = load_trimesh(file_path)
    return np.array(mesh.vertices), np.array(mesh.faces)


def _3mf_transform(attr: str | None) -> np.ndarray:
    """A 3MF ``transform`` attribute (12 numbers, row vectors) as a 4x4 matrix."""
    m = np.eye(4)
    if attr:
        v = [float(x) for x in attr.split()]
        m[:3, :3] = np.array(v[:9]).reshape(3, 3).T
        m[:3, 3] = v[9:12]
    return m


def read3MF(file_name) -> dict:
    """``{name: (vertices, faces)}`` of a 3MF's build items, placed by their transforms.

    The standard library's XML parser, streamed: trimesh's 3MF reader needs lxml, and
    ``write3MF`` above writes without it. Objects made of components are flattened, with
    the component transforms applied; one entry per build item, named after its object
    (``name`` attribute, else ``object<id>``; repeats get ``#n``).
    """
    import zipfile
    from xml.etree.ElementTree import iterparse

    with zipfile.ZipFile(file_name) as zf:
        model = next((n for n in zf.namelist() if n.lower().endswith(".model")), None)
        if model is None:
            raise ValueError(f"{file_name}: no 3D model in the package")
        objects: dict = {}      # id -> {"name", "v", "f", "components": [(id, matrix)]}
        items: list = []        # (object id, matrix)
        cur = None
        v_rows: list = []
        f_rows: list = []
        with zf.open(model) as fh:
            for event, el in iterparse(fh, events=("start", "end")):
                tag = el.tag.rsplit("}", 1)[-1]
                if event == "start" and tag == "object":
                    cur = {"name": el.get("name") or f"object{el.get('id')}", "components": []}
                    v_rows, f_rows = [], []
                    objects[el.get("id")] = cur
                elif event == "end":
                    if tag == "vertex":
                        v_rows.append((float(el.get("x")), float(el.get("y")), float(el.get("z"))))
                    elif tag == "triangle":
                        f_rows.append((int(el.get("v1")), int(el.get("v2")), int(el.get("v3"))))
                    elif tag == "component" and cur is not None:
                        cur["components"].append((el.get("objectid"), _3mf_transform(el.get("transform"))))
                    elif tag == "object" and cur is not None:
                        cur["v"] = np.array(v_rows, dtype=np.float64).reshape(-1, 3)
                        cur["f"] = np.array(f_rows, dtype=np.int64).reshape(-1, 3)
                        cur = None
                    elif tag == "item":
                        items.append((el.get("objectid"), _3mf_transform(el.get("transform"))))
                    if tag in ("vertex", "triangle", "component", "item"):
                        el.clear()

    def flatten(oid: str, m: np.ndarray, depth: int = 0):
        obj = objects[oid]
        vs, fs, n = [], [], 0
        if len(obj.get("f", ())):
            v = obj["v"] @ m[:3, :3].T + m[:3, 3]
            vs.append(v)
            fs.append(obj["f"] + n)
            n += len(v)
        if depth < 16:
            for cid, cm in obj["components"]:
                cv, cf = flatten(cid, m @ cm, depth + 1)
                if len(cf):
                    vs.append(cv)
                    fs.append(cf + n)
                    n += len(cv)
        if not vs:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        return np.vstack(vs), np.vstack(fs)

    out: dict = {}
    for oid, m in items or [(k, np.eye(4)) for k in objects]:
        v, f = flatten(oid, m)
        if not len(f):
            continue
        name = objects[oid]["name"]
        if name in out:
            name = f"{name}#{len(out)}"
        out[name] = (v, f)
    return out
