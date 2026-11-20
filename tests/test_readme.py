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


def test_spec_status_matches_the_index_001_AC10() -> None:
    spec = (ROOT / "specs" / "001-linux-viewer.md").read_text()
    status = (
        next(line for line in spec.splitlines() if line.startswith("**Status**:")).split(":", 1)[1].strip()
    )
    row = next(
        line
        for line in (ROOT / "specs" / "README.md").read_text().splitlines()
        if "001-linux-viewer.md" in line
    )
    assert row.rstrip(" |").rsplit("|", 1)[1].strip() == status
