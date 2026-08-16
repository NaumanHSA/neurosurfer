"""Callables for `function` nodes, in the three shapes the reads rule must tell apart.

A code node is invoked `fn(**{**graph_inputs, **dependency_results, **scope})`, so
its *signature* is the list of graph inputs it reads. These exist as a module rather
than as lambdas in the test because the rule resolves a node's `callable` by import
string, which a locally defined function has no path to.
"""

from __future__ import annotations


def reads_one(article: str, **_) -> str:
    """Names one input and sweeps up the rest — the ordinary shape."""
    return article[:10]


def reads_one_strictly(article: str) -> str:
    """Names one input and nothing else, so an extra really is unread."""
    return article[:10]


def reads_whatever_it_is_given(**kwargs) -> str:
    """Declares only `**kwargs`, so no input can be said to go unread."""
    return ", ".join(sorted(kwargs))
