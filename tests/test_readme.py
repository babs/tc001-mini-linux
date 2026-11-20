from pathlib import Path

ROOT = Path(__file__).parent.parent


def section(path: Path, heading: str) -> list[str]:
    """Lines of the `## heading` section, without the heading and the spec's README note."""
    lines = path.read_text().splitlines()
    start = lines.index(heading) + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    body = [line for line in lines[start:end] if not line.startswith("The README carries this section")]
    return [line for line in body if line.strip()]


def test_readme_usage_is_the_spec_usage_001_AC10() -> None:
    spec = section(ROOT / "specs" / "001-linux-viewer.md", "## Usage (README content)")
    readme = section(ROOT / "README.md", "## Usage")
    assert spec and readme == spec
