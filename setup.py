"""Setup script for rad-report-engine package."""

from pathlib import Path

from setuptools import find_packages, setup


def load_project_dependencies() -> list[str]:
    """Read dependencies directly from pyproject.toml to keep a single source of truth."""
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - only hit on <3.11
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("tomli is required to parse pyproject.toml on Python < 3.11") from exc

    pyproject_path = Path(__file__).parent / "pyproject.toml"
    if not pyproject_path.exists():
        return []

    with pyproject_path.open("rb") as fp:
        project_table = tomllib.load(fp).get("project", {})

    return project_table.get("dependencies", [])


# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

requirements = load_project_dependencies()

setup(
    name="rad-report-engine",
    version="1.0.0",
    author="YalaLab",
    author_email="",
    description="AI-powered radiology report processing and analysis engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yalalab/rad-report-engine",
    packages=["rate", "rate.core"],
    package_dir={"rate": "src", "rate.core": "src/core"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rad-report-engine=rate.cli:main",
            "rad-report-qc=rate.qc_cli:main",
            "rad-report-eval=rate.eval_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/modalities/*.yaml", "config/modalities/*.yml"],
    },
    zip_safe=False,
)
