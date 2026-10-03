"""numpy2stl.io.parts: models as named parts (F-STL2NUMPY, 2026-10-03)."""
import pytest
import trimesh

from numpy2stl.io import load_parts, part_role, read3MF, read_parts_file, write3MF, write_parts_file


def _box(at=0.0):
    return trimesh.creation.box(extents=(1, 1, 1)).apply_translation((at, 0, 0))


def test_a_pack_folder_keeps_each_file(tmp_path):
    """A Micropolitan pack: the solid and the water come as separate STLs."""
    _box(0).export(tmp_path / "Barcelona, Spain_L_Solid.stl")
    _box(3).export(tmp_path / "Barcelona, Spain_L_Water.stl")
    (tmp_path / "notes.txt").write_text("not a mesh")
    parts = load_parts(tmp_path)
    assert sorted(parts) == ["Barcelona, Spain_L_Solid", "Barcelona, Spain_L_Water"]
    assert {part_role(k) for k in parts} == {"model", "water"}


def test_a_3mf_keeps_its_objects_in_place(tmp_path):
    """numpy2stl's own 3MF (write3MF) read back without lxml: one part per object."""
    t, b = _box(0), _box(5)
    write3MF(str(tmp_path / "city.3mf"), {"terrain": (t.vertices, t.faces),
                                         "buildings": (b.vertices, b.faces)})
    parts = load_parts(tmp_path / "city.3mf")
    assert sorted(parts) == ["city:buildings", "city:terrain"]
    assert parts["city:buildings"].bounds[0][0] == pytest.approx(4.5)
    assert len(parts["city:terrain"].faces) == 12


def test_3mf_transforms_and_components(tmp_path):
    """Build-item and component transforms are applied (a slicer's 3MF moves objects)."""
    import zipfile
    xml = ('<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>'
           '<object id="1" name="tri"><mesh><vertices><vertex x="0" y="0" z="0"/>'
           '<vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/></vertices>'
           '<triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh></object>'
           '<object id="2" name="group"><components>'
           '<component objectid="1" transform="1 0 0 0 1 0 0 0 1 10 0 0"/></components></object>'
           '</resources><build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 0 0 5"/></build></model>')
    with zipfile.ZipFile(tmp_path / "t.3mf", "w") as zf:
        zf.writestr("3D/3dmodel.model", xml)
    (v, f), = read3MF(tmp_path / "t.3mf").values()
    assert v.min(axis=0).tolist() == [10, 0, 5] and f.tolist() == [[0, 1, 2]]


def test_parts_file_round_trip(tmp_path):
    write_parts_file({"terrain": _box(0), "roads": _box(2), "empty": trimesh.Trimesh()},
                     tmp_path / "m.parts")
    parts = load_parts(tmp_path / "m.parts")
    assert sorted(parts) == ["roads", "terrain"]
    assert len(read_parts_file(tmp_path / "m.parts")["roads"].faces) == 12


@pytest.mark.parametrize("name, role", [
    ("waterways", "water"), ("Valencia_L_Water", "water"), ("roads", "roads"),
    ("railways", "roads"), ("buildings", "buildings"), ("churches", "buildings"),
    ("terrain", "terrain"), ("Paris_XL_Solid", "model"), ("green", "other")])
def test_part_roles(name, role):
    assert part_role(name) == role


def test_load_trimesh_reads_3mf_without_lxml(tmp_path):
    from numpy2stl.io import load_trimesh
    t, b = _box(0), _box(5)
    write3MF(str(tmp_path / "m.3mf"), {"a": (t.vertices, t.faces), "b": (b.vertices, b.faces)})
    m = load_trimesh(tmp_path / "m.3mf")
    assert len(m.faces) == 24 and m.bounds[1][0] == pytest.approx(5.5)
