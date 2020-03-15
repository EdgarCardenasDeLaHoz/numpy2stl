from setuptools import setup, find_packages

setup(
    name="numpy2stl",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["shapely", "numpy"],
)
