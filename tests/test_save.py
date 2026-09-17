# Tests for save.py - Export functions
import numpy as np
from numpy2stl import writeSTL, write3MF, writeOBJ, triangles_to_facets


class TestWriteSTL:
    """Test STL file writing."""

    def test_binary_stl(self, cube_mesh, tmp_stl_file):
        """Test binary STL writing."""
        vertices, faces = cube_mesh
        triangles = vertices[faces]
        facets = triangles_to_facets(triangles)

        writeSTL(facets, str(tmp_stl_file), ascii=False)

        assert tmp_stl_file.exists()
        # Binary STL has fixed header + facet size
        # 80 bytes header + 4 bytes count + (50 bytes * num_facets)
        expected_size = 80 + 4 + (50 * len(facets))
        assert tmp_stl_file.stat().st_size == expected_size

    def test_ascii_stl(self, cube_mesh, tmp_stl_file):
        """Test ASCII STL writing."""
        vertices, faces = cube_mesh
        triangles = vertices[faces]
        facets = triangles_to_facets(triangles)

        writeSTL(facets, str(tmp_stl_file), ascii=True)

        assert tmp_stl_file.exists()
        content = tmp_stl_file.read_text()

        # Verify ASCII format
        assert content.startswith("solid ffd_geom")
        assert content.strip().endswith("endsolid ffd_geom")
        assert "facet normal" in content
        assert "vertex" in content

    def test_empty_facets(self, tmp_stl_file):
        """Test writing empty facet list."""
        facets = np.zeros((0, 12))

        writeSTL(facets, str(tmp_stl_file))

        assert tmp_stl_file.exists()
        # Should create minimal valid STL
        assert tmp_stl_file.stat().st_size > 0


class TestWrite3MF:
    """Test 3MF file writing."""

    def test_single_model(self, cube_mesh, tmp_3mf_file):
        """Test writing single model to 3MF."""
        models = {"cube": cube_mesh}

        write3MF(str(tmp_3mf_file), models)

        assert tmp_3mf_file.exists()
        assert tmp_3mf_file.stat().st_size > 0

        # 3MF is a zip file
        import zipfile

        with zipfile.ZipFile(tmp_3mf_file, "r") as zf:
            # Check required files exist
            assert "3D/3dmodel.model" in zf.namelist()
            assert "_rels/.rels" in zf.namelist()
            assert "[Content_Types].xml" in zf.namelist()

    def test_multiple_models(self, cube_mesh, simple_triangle_mesh, tmp_3mf_file):
        """Test writing multiple models to 3MF."""
        models = {"cube": cube_mesh, "triangle": simple_triangle_mesh}

        write3MF(str(tmp_3mf_file), models)

        assert tmp_3mf_file.exists()

        import zipfile
        import xml.etree.ElementTree as ET

        with zipfile.ZipFile(tmp_3mf_file, "r") as zf:
            model_xml = zf.read("3D/3dmodel.model").decode("utf-8")
            root = ET.fromstring(model_xml)

            # Should have 2 objects
            ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
            objects = root.findall(".//m:object", ns)
            assert len(objects) == 2

    def test_empty_models(self, tmp_3mf_file):
        """Test error handling for empty models dict."""
        models = {}

        # Should create file even with no models (valid 3MF structure)
        write3MF(str(tmp_3mf_file), models)

        assert tmp_3mf_file.exists()


class TestWriteOBJ:
    """Test OBJ file writing."""

    def test_single_model(self, cube_mesh, tmp_obj_file):
        """Test writing single model to OBJ."""
        models = {"cube": cube_mesh}

        writeOBJ(str(tmp_obj_file), models)

        assert tmp_obj_file.exists()
        content = tmp_obj_file.read_text()

        # Verify OBJ format
        assert "o cube" in content  # Object name
        assert "v " in content  # Vertices
        assert "f " in content  # Faces

        # Count vertices and faces
        vertex_lines = [l for l in content.split("\n") if l.startswith("v ")]
        face_lines = [l for l in content.split("\n") if l.startswith("f ")]

        assert len(vertex_lines) == 8  # Cube has 8 vertices
        assert len(face_lines) == 12  # Cube has 12 triangular faces

    def test_multiple_models(self, cube_mesh, simple_triangle_mesh, tmp_obj_file):
        """Test writing multiple models to OBJ."""
        models = {"cube": cube_mesh, "triangle": simple_triangle_mesh}

        writeOBJ(str(tmp_obj_file), models)

        content = tmp_obj_file.read_text()

        # Both object names should be present
        assert "o cube" in content
        assert "o triangle" in content

    def test_vertex_indexing(self, tmp_obj_file):
        """Test that OBJ indices are 1-based and cumulative."""
        # Create two simple models
        mesh1 = (
            np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]]),
            np.array([[0, 1, 2]]),
        )
        mesh2 = (
            np.array([[0, 0, 1], [1, 0, 1], [0.5, 1, 1]]),
            np.array([[0, 1, 2]]),
        )

        models = {"tri1": mesh1, "tri2": mesh2}
        writeOBJ(str(tmp_obj_file), models)

        content = tmp_obj_file.read_text()
        lines = content.split("\n")

        # Find face lines
        face_lines = [l for l in lines if l.startswith("f ")]

        # First face should use indices 1, 2, 3
        assert "f 1 2 3" in face_lines[0]

        # Second face should use indices 4, 5, 6 (cumulative)
        assert "f 4 5 6" in face_lines[1]
