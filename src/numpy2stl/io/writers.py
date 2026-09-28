import struct
import zipfile
from collections.abc import Iterator

__all__ = ["writeSTL", "write3MF", "writeOBJ"]


def _build_binary_stl(facets):
    """returns a string of binary binary data for the stl file"""

    BINARY_HEADER = "80sI"
    BINARY_FACET = "12fH"

    lines = [
        struct.pack(BINARY_HEADER, b"Binary STL Writer", len(facets)),
    ]
    for facet in facets:
        facet = list(facet)
        facet.append(0)  # need to pad the end with a unsigned short byte
        lines.append(struct.pack(BINARY_FACET, *facet))
    return lines


def _build_ascii_stl(facets):
    """returns a list of ascii lines for the stl file"""

    ASCII_FACET = """  facet normal  {face[0]:e}  {face[1]:e}  {face[2]:e}
        outer loop
        vertex    {face[3]:e}  {face[4]:e}  {face[5]:e}
        vertex    {face[6]:e}  {face[7]:e}  {face[8]:e}
        vertex    {face[9]:e}  {face[10]:e}  {face[11]:e}
        endloop
    endfacet"""

    lines = [
        "solid ffd_geom",
    ]
    for facet in facets:
        lines.append(ASCII_FACET.format(face=facet))
    lines.append("endsolid ffd_geom")
    return lines


def writeSTL(facets, file_name, ascii=False):
    """writes an ASCII or binary STL file"""

    f = open(file_name, "wb")
    if ascii:
        lines = _build_ascii_stl(facets)
        lines_ = "\n".join(lines).encode("UTF-8")
        f.write(lines_)
    else:
        data = _build_binary_stl(facets)
        data = b"".join(data)
        f.write(data)

    f.close()


_3MF_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_3MF_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
    '<Relationship Target="/3D/3dmodel.model" Id="rel1" '
    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />\n'
    "</Relationships>"
)
_3MF_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />\n'
    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />\n'
    "</Types>"
)
_3MF_CHUNK = 200_000   # rows formatted per write


def _rows(fmt: str, arr) -> "Iterator[bytes]":
    """``fmt % row`` for each row of ``arr``, in encoded chunks."""
    import numpy as np

    a = np.asarray(arr)
    for i in range(0, len(a), _3MF_CHUNK):
        yield "".join(map(fmt.__mod__, map(tuple, a[i:i + _3MF_CHUNK].tolist()))).encode()


def write3MF(file_name, models, compresslevel: int = 1):
    """Write ``{name: (vertices, faces)}`` as one 3MF package, one object per entry.

    The model XML is streamed into the zip in chunks with plain string
    formatting (vertices to 0.1 um): building an ElementTree node per vertex and
    triangle took ~30 s for a 3 M-triangle city model. ``compresslevel`` is
    zlib's (1 = fastest; the XML still shrinks ~4x).
    """
    from xml.sax.saxutils import quoteattr

    with zipfile.ZipFile(file_name, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=compresslevel) as zf:
        # 1. The Model
        with zf.open("3D/3dmodel.model", "w", force_zip64=True) as out:
            out.write(("<?xml version='1.0' encoding='utf-8'?>\n"
                       f'<model unit="millimeter" xml:lang="en-US" xmlns="{_3MF_NS}">'
                       "<resources>").encode())
            for i, (name, (vertices, faces)) in enumerate(models.items(), start=1):
                out.write(f'<object id="{i}" name={quoteattr(str(name))} type="model">'
                          "<mesh><vertices>".encode())
                for chunk in _rows('<vertex x="%.4f" y="%.4f" z="%.4f" />', vertices):
                    out.write(chunk)
                out.write(b"</vertices><triangles>")
                for chunk in _rows('<triangle v1="%d" v2="%d" v3="%d" />', faces):
                    out.write(chunk)
                out.write(b"</triangles></mesh></object>")
            out.write(b"</resources><build>")
            out.write("".join(f'<item objectid="{i}" />'
                              for i in range(1, len(models) + 1)).encode())
            out.write(b"</build></model>")
        # 2. Relationships, 3. Content types (both needed by strict readers)
        zf.writestr("_rels/.rels", _3MF_RELS)
        zf.writestr("[Content_Types].xml", _3MF_CONTENT_TYPES)

    import logging as _logging

    _logging.getLogger(__name__).info("Successfully saved %d objects to %s", len(models), file_name)


def writeOBJ(file_name, models):
    """
    Writes multiple meshes into a single OBJ file.
    Each key in the puzzle dictionary becomes a named object.
    """
    with open(file_name, "w") as f:
        f.write("# Exported Puzzle Project\n")

        v_offset = 1  # OBJ indices are 1-based and cumulative

        for key, (vertices, faces) in models.items():
            f.write(f"\no {key}\n")  # Define a new object

            # Write vertices for this object
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (shifting indices by the current offset)
            for face in faces:
                # OBJ indices: v1 v2 v3
                f.write(f"f {face[0] + v_offset} {face[1] + v_offset} {face[2] + v_offset}\n")

            # Update offset for the next object
            v_offset += len(vertices)

    import logging as _logging

    _logging.getLogger(__name__).info("Successfully saved %d objects to %s", len(models), file_name)
