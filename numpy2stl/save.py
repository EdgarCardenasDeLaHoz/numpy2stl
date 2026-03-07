
import struct

import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO

import xml.etree.ElementTree as ET
import zipfile

def _build_binary_stl(facets):
    """returns a string of binary binary data for the stl file"""

    BINARY_HEADER = "80sI"
    BINARY_FACET = "12fH"

    lines = [struct.pack(BINARY_HEADER, b'Binary STL Writer', len(facets)), ]
    for facet in facets:
        facet = list(facet)
        facet.append(0)  # need to pad the end with a unsigned short byte
        lines.append(struct.pack(BINARY_FACET, *facet))
    return lines

def _build_ascii_stl(facets):
    """returns a list of ascii lines for the stl file """

    ASCII_FACET = """  facet normal  {face[0]:e}  {face[1]:e}  {face[2]:e}
        outer loop
        vertex    {face[3]:e}  {face[4]:e}  {face[5]:e}
        vertex    {face[6]:e}  {face[7]:e}  {face[8]:e}
        vertex    {face[9]:e}  {face[10]:e}  {face[11]:e}
        endloop
    endfacet"""

    lines = ['solid ffd_geom', ]
    for facet in facets:
        lines.append(ASCII_FACET.format(face=facet))
    lines.append('endsolid ffd_geom')
    return lines

def writeSTL(facets, file_name, ascii=False):
    """writes an ASCII or binary STL file"""

    f = open(file_name, 'wb')
    if ascii:
        lines = _build_ascii_stl(facets)
        lines_ = "\n".join(lines).encode("UTF-8")
        f.write(lines_)
    else:
        data = _build_binary_stl(facets)
        data = b"".join(data)
        f.write(data)

    f.close()



def write3MF(file_name, models):
    # Create the root <model> element
    root = ET.Element('model', {
        'unit': 'millimeter',
        'xml:lang': 'en-US',
        'xmlns': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'
    })
    
    resources = ET.SubElement(root, 'resources')
    build = ET.SubElement(root, 'build')
    
    for i, (name, (vertices, faces)) in enumerate(models.items(), start=1):
        obj = ET.SubElement(resources, 'object', {'id': str(i), 'name': name, 'type': 'model'})
        mesh = ET.SubElement(obj, 'mesh')
        
        verts_elem = ET.SubElement(mesh, 'vertices')
        for v in vertices:
            ET.SubElement(verts_elem, 'vertex', {
                'x': f"{v[0]:.4f}", 'y': f"{v[1]:.4f}", 'z': f"{v[2]:.4f}"
            })
        
        triangles_elem = ET.SubElement(mesh, 'triangles')
        for f in faces:
            ET.SubElement(triangles_elem, 'triangle', {
                'v1': str(f[0]), 'v2': str(f[1]), 'v3': str(f[2])
            })
            
        ET.SubElement(build, 'item', {'objectid': str(i)})

    xml_content = ET.tostring(root, encoding='utf-8', xml_declaration=True)

    with zipfile.ZipFile(file_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. The Model
        zf.writestr('3D/3dmodel.model', xml_content)
        
        # 2. Corrected Relationships
        rels_content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            '<Relationship Target="/3D/3dmodel.model" Id="rel1" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />\n'
            '</Relationships>'
        )
        zf.writestr('_rels/.rels', rels_content)

        # 3. Content Types (Crucial for full compatibility)
        content_types = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />\n'
            '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />\n'
            '</Types>'
        )
        zf.writestr('[Content_Types].xml', content_types)

    print(f"✅ Successfully saved {len(models)} objects to {file_name}")

def writeOBJ(file_name, models):
    """
    Writes multiple meshes into a single OBJ file.
    Each key in the puzzle dictionary becomes a named object.
    """
    with open(file_name, 'w') as f:
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

    print(f"✅ Successfully saved {len(models)} objects to {file_name}")