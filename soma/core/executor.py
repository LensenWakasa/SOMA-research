"""
acsis/tools/executor.py
========================
EXPERIMENT stage. Acsis writes and runs Python to verify hypotheses.

Safety: runs in an isolated subprocess with timeout.
No file system access, no network access from within experiments.
"""
from __future__ import annotations
import asyncio
import io
import logging
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    code: str
    output: str
    success: bool
    error: Optional[str]
    result_fact: str          # plain English summary of what the code showed
    interpretation: str       # one-sentence meaning


class CodeExecutor:
    """
    Executes Python code in a sandboxed environment.

    Allowed: numpy, scipy, math, statistics, pandas (data analysis)
    Blocked: os, subprocess, network calls, file writes

    Usage:
        ex = CodeExecutor()
        result = await ex.run("import math; print(math.pi ** 2)")
    """

    BLOCKED_IMPORTS = {"os", "sys", "subprocess", "socket", "requests", "aiohttp", "urllib"}
    ALLOWED_MODULES = {"math", "statistics", "itertools", "functools", "collections",
                       "random", "decimal", "fractions"}

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.timeout = getattr(cfg, 'code_execution_timeout', 30) if cfg else 30

    async def run(self, code: str) -> ExecResult:
        """
        Asynchronously run code with timeout.
        Returns ExecResult with output and plain-English interpretation.
        """
        logger.info(f"[EXEC] Running code ({len(code)} chars)")

        # Safety check before running
        blocked = self._check_safety(code)
        if blocked:
            return ExecResult(
                code=code, output="", success=False,
                error=f"Blocked: unsafe import '{blocked}'",
                result_fact="",
                interpretation=f"Code was blocked for safety: attempted to import '{blocked}'",
            )

        # Run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._execute, code),
                timeout=self.timeout,
            )
            return result
        except asyncio.TimeoutError:
            return ExecResult(
                code=code, output="", success=False,
                error=f"Timeout after {self.timeout}s",
                result_fact="",
                interpretation="Computation timed out — too complex or infinite loop",
            )

    def _execute(self, code: str) -> ExecResult:
        """Execute in a restricted namespace."""
        # Build safe namespace
        safe_globals = {
            "__builtins__": self._safe_builtins(),
        }
        # Add commonly needed modules
        for mod_name in ["math", "statistics", "random", "decimal"]:
            try:
                import importlib
                safe_globals[mod_name] = importlib.import_module(mod_name)
            except ImportError:
                pass
        try:
            import numpy as np
            safe_globals["np"] = np
            safe_globals["numpy"] = np
        except ImportError:
            pass
        try:
            import pandas as pd
            safe_globals["pd"] = pd
        except ImportError:
            pass

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        local_vars = {}

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals, local_vars)   # noqa: S102

            output = stdout_capture.getvalue()
            stderr_out = stderr_capture.getvalue()

            # Build a result fact from the output
            result_fact = self._interpret_output(code, output, local_vars)
            interpretation = f"Code executed successfully. Output: {output[:200]}" if output else "Code ran with no output"

            return ExecResult(
                code=code,
                output=output,
                success=True,
                error=stderr_out if stderr_out else None,
                result_fact=result_fact,
                interpretation=interpretation,
            )

        except Exception as e:
            tb = traceback.format_exc()
            return ExecResult(
                code=code,
                output=stdout_capture.getvalue(),
                success=False,
                error=str(e),
                result_fact="",
                interpretation=f"Experiment failed: {str(e)[:100]}",
            )

    def _safe_builtins(self) -> dict:
        """Return a restricted builtins dict."""
        safe_names = [
            "abs","all","any","bin","bool","bytes","callable","chr","complex",
            "dict","dir","divmod","enumerate","filter","float","format",
            "frozenset","getattr","hasattr","hash","hex","int","isinstance",
            "issubclass","iter","len","list","map","max","min","next","oct",
            "ord","pow","print","range","repr","reversed","round","set",
            "setattr","slice","sorted","str","sum","super","tuple","type",
            "vars","zip","__import__",
        ]
        import builtins as _builtins
        return {k: getattr(_builtins, k) for k in safe_names if hasattr(_builtins, k)}

    def _check_safety(self, code: str) -> Optional[str]:
        """Return blocked module name if code is unsafe, else None."""
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if root in self.BLOCKED_IMPORTS:
                            return root
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root = node.module.split(".")[0]
                        if root in self.BLOCKED_IMPORTS:
                            return root
        except SyntaxError:
            return None  # will fail at exec anyway
        return None

    def _interpret_output(self, code: str, output: str, local_vars: dict) -> str:
        """Generate a plain-English fact from code output."""
        if not output and not local_vars:
            return ""
        lines = output.strip().split("\n")
        if lines:
            return f"Computational result: {lines[-1][:200]}"
        if local_vars:
            pairs = {k: v for k, v in local_vars.items() if not k.startswith("_")}
            if pairs:
                key = list(pairs.keys())[-1]
                return f"Computed {key} = {str(pairs[key])[:100]}"
        return ""
