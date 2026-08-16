"""Test-session setup shared by every test.

Currently one thing: making `caplog` able to see the library's own log records.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _let_caplog_see_neurosurfer_logs():
    """Let `caplog` capture records from the `neurosurfer` logger tree.

    `configure_logging` attaches a Rich handler to the `neurosurfer` logger and
    sets `propagate = False`, which is correct for a library: a package that owns
    a handler should not also emit through the root logger and print everything
    twice. It runs on the first `get_logger` call, so it has happened before any
    test body starts.

    `caplog`, though, captures at the **root**. With propagation off, records stop
    one logger short of it and `caplog.records` is empty however loudly the code
    logged — the message is right there in the captured stdout, and the assertion
    about it fails. Eight tests across observability and the engine were written
    against that gap and could not pass.

    Turning propagation back on for the duration of a test is the smallest fix
    that keeps production behaviour intact: the handler stays where it is, the
    library still logs exactly once outside the suite, and `caplog` sees what it
    was always meant to see.
    """
    logger = logging.getLogger("neurosurfer")
    previous = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = previous
