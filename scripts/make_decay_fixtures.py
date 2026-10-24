"""The decay-time fixture: pyslfp's estimator on a deterministic noisy
series, for TestMeasurements to hold the C++ port to.

Run in this directory's environment (pyproject.toml):

    cd scripts && poetry install --no-root
    poetry run python make_decay_fixtures.py [--data-dir DIR]

The series is a clean decay plus reproducible noise from the same
linear-congruential generator the C++ test uses elsewhere, fitted with
the times pinned and with them free; the fixture holds the inputs, the
fitted decay time, amplitude and offset, and every weight, floats
written to round-trip.
"""

import numpy as np
from pyslfp.viscoelastic import fit_decay_time

from fixture_tools import data_dir


def series() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 4.0, 10)
    y = 25.0 * np.exp(-t / 1.1) + 2.0
    state = 123456789
    for j in range(y.size):
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        y[j] += 0.4 * ((state >> 11) / 9007199254740992.0 - 0.5)
    return t, y


def main() -> None:
    out = data_dir(__doc__) / "decay_time.txt"
    t, y = series()
    lines = []
    for name, sigma_t in (("pinned", 0.0), ("free", 0.05)):
        fit = fit_decay_time(t, y, time_uncertainty=sigma_t,
                             value_uncertainty=0.3)
        lines.append(f"case {name} {sigma_t!r} 0.3")
        lines += [f"sample {float(a)!r} {float(b)!r}" for a, b in zip(t, y)]
        lines.append(f"decay_time {float(fit.decay_time)!r}")
        lines.append(f"amplitude {float(fit.amplitude)!r}")
        lines.append(f"offset {float(fit.offset)!r}")
        lines += [f"weight {j} {float(w)!r}" for j, w in enumerate(fit.weights)]
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
