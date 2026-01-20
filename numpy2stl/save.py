
import struct

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