"""neurosurfer CLI.

No subcommand  → interactive REPL (banner, status, slash commands, chat).
Subcommands (scriptable):
  doctor                       config + active-connection reachability
  task list | show <name>      task registry
  provider list | use <name> | add | delete <name>
  run <task> [k=v ...]         run a task non-interactively
  serve [--host] [--port] ...  start the OpenAI-compatible gateway
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from neurosurfer.config import Config, load_config
from neurosurfer.observability.logging import configure_logging
from .context import CLIContext

# Register the coding-assistant personas (explore/analyzer/writer/verifier) into
# the engine registry. The framework engine ships no personas; the product layer
# wires them in at startup so spawn_agent can resolve them.
import neurosurfer.app  # noqa: F401,E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neurosurfer",
        description="Build intelligent apps that blend LLM reasoning, tools, and retrieval — CPU-only or GPU-accelerated.",
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Check configuration and the active connection")

    task_p = sub.add_parser("task", help="Manage Tasks")
    task_sub = task_p.add_subparsers(dest="task_command")
    task_sub.add_parser("list", help="List registered Tasks")
    show_p = task_sub.add_parser("show", help="Show a Task definition")
    show_p.add_argument("name")

    prov_p = sub.add_parser("provider", help="Manage provider profiles")
    prov_sub = prov_p.add_subparsers(dest="provider_command")
    prov_sub.add_parser("list", help="List provider profiles")
    use_p = prov_sub.add_parser("use", help="Set the active provider profile")
    use_p.add_argument("name")
    prov_sub.add_parser("add", help="Add a provider profile (interactive)")
    del_p = prov_sub.add_parser("delete", help="Delete a provider profile")
    del_p.add_argument("name")

    run_p = sub.add_parser("run", help="Run a registered Task (non-interactive)")
    run_p.add_argument("task")
    run_p.add_argument("inputs", nargs="*", help="key=value inputs")

    from .commands.serve import add_serve_parser

    add_serve_parser(sub)

    return p


def _run_async_command(coro) -> int:
    asyncio.run(coro)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)

    cfg: Config = load_config()
    configure_logging(cfg.observability.log_level)
    ctx = CLIContext.create(cfg)

    if args.command is None:
        from .app import run_repl

        return asyncio.run(run_repl(cfg))

    if args.command == "doctor":
        from .doctor import cmd_doctor

        return cmd_doctor(ctx)

    if args.command == "task":
        from .commands.task import handle as task_handle

        sub_args = [args.task_command] if args.task_command else []
        if args.task_command == "show":
            sub_args.append(args.name)
        return _run_async_command(task_handle(ctx, sub_args))

    if args.command == "provider":
        from .commands.provider import handle as provider_handle

        sub_args = [args.provider_command] if args.provider_command else []
        if args.provider_command in ("use", "delete"):
            sub_args.append(args.name)
        return _run_async_command(provider_handle(ctx, sub_args))

    if args.command == "run":
        from .commands.task import handle_run

        return _run_async_command(handle_run(ctx, [args.task, *args.inputs]))

    if args.command == "serve":
        from .commands.serve import handle_serve

        return handle_serve(args)

    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
