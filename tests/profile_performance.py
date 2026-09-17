"""Profile numpy2stl performance to identify bottlenecks."""
import cProfile
import pstats
from io import StringIO
import numpy as np
from numpy2stl import array_to_mesh


def profile_array_to_mesh():
    """Profile array_to_mesh with different array sizes."""
    sizes = [100, 500, 1000]

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Profiling array_to_mesh with {size}x{size} array")
        print(f"{'='*60}\n")

        # Generate test data
        np.random.seed(42)
        arr = np.random.rand(size, size) * 100

        # Profile
        profiler = cProfile.Profile()
        profiler.enable()

        vertices, faces = array_to_mesh(arr, solid=True)

        profiler.disable()

        # Print stats
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Top 20 functions
        print(s.getvalue())

        print(f"Result: {len(vertices)} vertices, {len(faces)} faces")


def profile_vertices_to_index():
    """Profile vertices_to_index function."""
    from numpy2stl import vertices_to_index

    print(f"\n{'='*60}")
    print("Profiling vertices_to_index")
    print(f"{'='*60}\n")

    # Generate test data
    np.random.seed(42)
    triangles = np.random.rand(50000, 3, 3) * 100

    # Profile
    profiler = cProfile.Profile()
    profiler.enable()

    vertices, faces = vertices_to_index(triangles)

    profiler.disable()

    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    print(f"Result: {len(vertices)} unique vertices from {len(triangles)} triangles")


if __name__ == "__main__":
    print("Starting performance profiling...")
    profile_array_to_mesh()
    profile_vertices_to_index()
    print("\nProfileng complete!")
