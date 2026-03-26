# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General

- Always line break documents and plans at 109 columns to keep them readable on small panes

## Commands

- **Check (format, lint, spell, type-check):** `just check`
- **Auto-format:** `just fmt`
- **Run all tests:** `just test`
- **Run specific tests:** `python3 -m barrage tests/test_framework.py` or `python3 -m barrage tests/test_framework.py::TestClassName::test_method`
- **Run mypy only:** `mypy ./barrage ./tests`

## Architecture

Barrage is a concurrent async test framework for Python 3.12+ with zero external dependencies. It uses itself to run its own tests.

## Python guidelines

- Don't use `TypeVar`, use the new generics syntax added in python 3.12.
- When setting up something that needs to be cleaned up, always use context managers to ensure proper cleanup. Use ExitStack or AsyncExitStack where needed to manage context managers.
- Never use `from __future__ import annotations`
- Always use `Self` to refer to a class's type within its definition instead of string types.

## Commit guidelines

- Always do Signed off by for commits
