"""Move sl3dpp's pins on its sibling libraries.

The SL3DPP_<NAME>_TAG defaults in CMakeLists.txt pin each library to a
commit. This moves them, every library to its current main by default, or
the ones named to a branch, tag or commit:

    python3 scripts/bump_dependencies.py
    python3 scripts/bump_dependencies.py planetmodel=v1.3.2 GSHTrans=main

and prints what moved. The pin is always the full commit hash; a tag is
recorded beside it, in the cache variable's description. Build and test
against the result (or let CI do so) before committing it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

LIBRARIES = [
    "NumericConcepts",
    "GaussQuad",
    "Interpolation",
    "FFTWpp",
    "GSHTrans",
    "planetmodel",
]

CMAKELISTS = Path(__file__).resolve().parent.parent / "CMakeLists.txt"


def resolve(library: str, revision: str) -> tuple[str, str | None]:
    """The commit a revision names, and the tag when it is one."""
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision, None
    url = f"https://github.com/da380/{library}.git"
    refs = subprocess.run(
        ["git", "ls-remote", url, f"refs/heads/{revision}",
         f"refs/tags/{revision}", f"refs/tags/{revision}^{{}}"],
        check=True, capture_output=True, text=True).stdout.split("\n")
    found = dict(line.split("\t")[::-1] for line in refs if line)
    # an annotated tag's own object is not the commit; ^{} peels it
    for ref, tag in [(f"refs/tags/{revision}^{{}}", revision),
                     (f"refs/tags/{revision}", revision),
                     (f"refs/heads/{revision}", None)]:
        if ref in found:
            return found[ref], tag
    sys.exit(f"{library}: no branch or tag {revision!r} at {url}")


def main(arguments: list[str]) -> None:
    wanted = dict.fromkeys(LIBRARIES, "main") if not arguments else {}
    for argument in arguments:
        library, _, revision = argument.partition("=")
        if library not in LIBRARIES or not revision:
            sys.exit(f"expected <library>=<revision> with a library of "
                     f"{', '.join(LIBRARIES)}; got {argument!r}")
        wanted[library] = revision

    text = CMAKELISTS.read_text()
    for library, revision in wanted.items():
        commit, tag = resolve(library, revision)
        pattern = re.compile(
            rf'(set\(SL3DPP_{library.upper()}_TAG ")([0-9a-f]{{40}})("\s*'
            rf'CACHE STRING ")([^"]*)(")')
        match = pattern.search(text)
        if not match:
            sys.exit(f"{library}: no SL3DPP_{library.upper()}_TAG pin found "
                     f"in {CMAKELISTS}")
        description = f"{library} commit" + (f" ({tag})" if tag else "")
        text = (text[:match.start()]
                + match.expand(rf"\g<1>{commit}\g<3>{description}\g<5>")
                + text[match.end():])
        moved = "unchanged" if match.group(2) == commit else \
            f"{match.group(2)[:7]} -> {commit[:7]}"
        print(f"{library:16} {revision:12} {moved}")
    CMAKELISTS.write_text(text)


if __name__ == "__main__":
    main(sys.argv[1:])
