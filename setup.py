import os

from setuptools import find_packages, setup

# Read README if it exists
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="numpy2stl",
    version="0.1.0",
    packages=["numpy2stl"],
    package_dir={"numpy2stl": "."},
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",  # Python 3.11+ compatibility
        "shapely>=2.0.0",  # Modern shapely with better performance
    ],
    extras_require={
        "core": [
            "scipy>=1.9.0",  # For generate.py (ndimage)
            "triangle>=20200424",  # For polygon triangulation
        ],
        "tools": [
            "opencv-python>=4.5.0",  # For tools.py (resize, rescale)
            "scikit-image>=0.19.0",  # For tools.py (filters)
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "napari[all]>=0.4.0",  # Optional, heavy dependency
        ],
        "boolean": [
            "pymeshlab>=2022.2",  # For boolean operations
            "manifold3d>=2.4.0",  # Alternative boolean engine
        ],
        "geo": [
            "rasterio>=1.3.0",
            "h5py>=3.7.0",
        ],
        "validation": [
            "trimesh>=3.9.0",  # For verify.py
        ],
        "puzzle": [
            "trimesh>=3.9.0",  # For puzzle.py
        ],
        "simplify": [
            "scipy>=1.9.0",  # For simplify.py
            "triangle>=20200424",
        ],
        "registration": [
            "osmnx>=2.0.0",
            "geopandas>=1.0.0",
            "rasterio>=1.3.0",
            "opencv-python>=4.5.0",
            "scikit-image>=0.19.0",
            "scipy>=1.9.0",
        ],
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.0.250",
            "mypy>=1.0.0",
        ],
    },
    author="Edgar Cardenas De La Hoz",
    description="Convert NumPy arrays and geometry into STL/OBJ/3MF meshes for 3D printing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EdgarCardenasDeLaHoz/numpy2stl",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
