"""What the fixture scripts share: where the fixtures are read and written,
and the provenance a generated fixture records."""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import pathlib

#: The committed fixtures, which the scripts read and write by default.
DATA = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data"


def parser(description: str | None) -> argparse.ArgumentParser:
    """An argument parser offering --data-dir: the directory a script reads
    its inputs from and writes its fixtures to. The default is tests/data;
    a scratch directory holding copies of the inputs regenerates the
    fixtures for comparison without touching the committed ones."""
    out = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    out.add_argument("--data-dir", type=pathlib.Path, default=DATA,
                     help="where the inputs are read and the fixtures "
                          "written (default: tests/data)")
    return out


def data_dir(description: str | None) -> pathlib.Path:
    """The --data-dir of a script that takes no other arguments."""
    directory = parser(description).parse_args().data_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def provenance(*packages: str) -> str:
    """The installed versions of the packages and today's date, as a
    fixture's header records them: 'pyslfp 2.2.2, planetmodel 1.3.2,
    2026-10-23'."""
    versions = [f"{p} {importlib.metadata.version(p)}" for p in packages]
    return ", ".join(versions + [datetime.date.today().isoformat()])
