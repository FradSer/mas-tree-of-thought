"""Test session setup.

Loads the project's ``dialectica/.env`` so the ``e2e`` skip guard can see
``GOOGLE_API_KEY`` (whether it comes from the real environment or ``.env``).
The library itself does not load ``.env`` — that is a test/app concern.

Also applies compatibility patches for third-party deprecation warnings that
we cannot fix upstream:

- **pytest-bdd 8.1** calls ``_register_fixture`` with the deprecated
  ``nodeid`` parameter (two warnings per call: ``nodeid`` + ``baseid``).
  We patch ``inject_fixture`` in both ``pytest_bdd.compat`` and the
  ``pytest_bdd.scenario`` module so it passes ``node`` instead.
- **ADK 2.6.x** emits ``BaseAgentConfig`` deprecation warnings (frozen by
  ADK's own constraint on ``opentelemetry-api<1.43``).
- **OpenTelemetry 1.42.x** emits ``SelectableGroups`` deprecation warnings.

Mock helpers live in ``tests/helpers.py``.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(
    Path(__file__).resolve().parent.parent / "dialectica" / ".env", override=True
)

# ---------------------------------------------------------------------------
# Monkey-patch pytest-bdd's inject_fixture to use ``node`` instead of the
# deprecated ``nodeid`` parameter.  pytest 9.1 added ``node`` while
# deprecating ``nodeid``; both are accepted, so this is a safe forward-port
# that avoids ~260 ``PytestRemovedIn10Warning`` messages.
#
# We must patch the function object in *both* locations:
#   - ``pytest_bdd.compat`` (where it is defined)
#   - ``sys.modules["pytest_bdd.scenario"]`` (where it is imported via
#     ``from .compat import inject_fixture``, creating a local binding)
# ---------------------------------------------------------------------------
import pytest_bdd.compat as _pytest_bdd_compat


def _patched_inject_fixture(request, arg, value):
    request._fixturemanager._register_fixture(
        name=arg,
        func=lambda: value,
        node=request.node,
    )
    fixture_def = request._get_active_fixturedef(arg)
    fixture_def.cached_result = (value, None, None)


_pytest_bdd_compat.inject_fixture = _patched_inject_fixture

_scenario_mod = sys.modules.get("pytest_bdd.scenario")
if _scenario_mod is not None:
    _scenario_mod.inject_fixture = _patched_inject_fixture
