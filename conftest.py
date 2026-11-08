"""Test-session path hygiene: keep foreign interpreters' site-packages off sys.path.

The Hermes desktop runtime exports PYTHONPATH pointing at its own virtualenv
(for example ``.../lib/python3.14/site-packages``). When this project's venv
runs a different interpreter (here 3.11), that leaked path precedes the
project's own site-packages, and the compiled extensions it finds there are
built for the wrong ABI — so importing them raises::

    ImportError: cannot import name 'ArgsKwargs' from 'pydantic_core._pydantic_core'

The failure is an environment leak, not a packaging defect: the project's own
``pydantic`` / ``pydantic-core`` pair is consistent. This guard drops any
``site-packages`` entry whose ``pythonX.Y`` tag does not match the running
interpreter before pytest collects, so the suite is runnable from a shell that
inherits the Hermes PYTHONPATH. It only edits ``sys.path``; it never patches or
reinstalls the packages themselves.
"""

from __future__ import annotations

import re
import sys

_SITE_PACKAGES_TAG = re.compile(r"lib/python(\d+)\.(\d+)/site-packages")


def _foreign_site_packages() -> list[str]:
    running = sys.version_info[:2]
    foreign: list[str] = []
    for entry in sys.path:
        match = _SITE_PACKAGES_TAG.search(entry)
        if match and (int(match.group(1)), int(match.group(2))) != running:
            foreign.append(entry)
    return foreign


for _entry in _foreign_site_packages():
    while _entry in sys.path:
        sys.path.remove(_entry)
