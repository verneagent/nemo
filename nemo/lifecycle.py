"""Self-restart and upgrade helpers for the Nemo host process."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution


PACKAGE_NAME = "captain-nemo"

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RestartSpec:
  """Arguments needed to launch a replacement Nemo process."""

  chat_id: str
  project_dir: str
  agent: str
  model: str
  permission_mode: str
  effort: str = ""


@dataclass(frozen=True)
class CommandResult:
  """Small subprocess result safe to show in chat/logs."""

  returncode: int
  output: str


@dataclass(frozen=True)
class UpgradeCheckResult:
  """Result of checking PyPI for a newer Nemo release."""

  returncode: int
  current_version: str
  latest_version: str
  update_available: bool
  output: str


def is_editable_install(package: str = PACKAGE_NAME) -> bool:
  """Return True when ``package`` was installed with pip editable mode."""
  try:
    dist = distribution(package)
  except PackageNotFoundError:
    return False
  direct_url = dist.read_text("direct_url.json")
  if not direct_url:
    return False
  try:
    parsed = json.loads(direct_url)
  except json.JSONDecodeError:
    return False
  if not isinstance(parsed, dict):
    return False
  dir_info = parsed.get("dir_info")
  if not isinstance(dir_info, dict):
    return False
  return dir_info.get("editable") is True


def build_restart_args(spec: RestartSpec, original_args: list[str] | None = None) -> list[str]:
  """Build ``python -m nemo`` args for a replacement process.

  Preserve process-level options such as ``--profile``, ``--verbose``, and
  ``--system-prompt-file`` from the original invocation, while forcing the
  resolved chat/project/agent/model values so restarts never auto-discover a
  different chat.
  """
  args = list(original_args if original_args is not None else sys.argv[1:])
  skip_value_for = {
    "--chat-id",
    "--chat-name",
    "--project-dir",
    "--agent",
    "--model",
    "--effort",
    "--permission-mode",
  }
  out: list[str] = []
  i = 0
  while i < len(args):
    arg = args[i]
    if arg in skip_value_for:
      i += 2
      continue
    if any(arg.startswith(f"{flag}=") for flag in skip_value_for):
      i += 1
      continue
    out.append(arg)
    i += 1

  out.extend([
    "--chat-id", spec.chat_id,
    "--project-dir", spec.project_dir,
    "--agent", spec.agent,
    "--model", spec.model,
    "--permission-mode", spec.permission_mode,
  ])
  if spec.effort:
    out.extend(["--effort", spec.effort])
  return out


def helper_log_path(parent_pid: int) -> str:
  from .config import CONFIG_DIR

  log_dir = os.path.join(CONFIG_DIR, "logs")
  os.makedirs(log_dir, exist_ok=True)
  return os.path.join(log_dir, f"nemo-lifecycle-{parent_pid}.log")


# Env vars that must keep the running daemon's own values on restart. A
# freshly sourced shell may rebuild PATH from its profile (brew, pyenv, nvm
# shims, …) or carry per-invocation noise (PWD/SHLVL/_); those are never
# overlaid — only profile-defined credential/setting vars are.
_PROTECTED_RESTART_ENV: frozenset[str] = frozenset({
  "HOME", "PATH", "PWD", "OLDPWD", "SHLVL", "_", "TMPDIR", "TERM",
  "TERM_PROGRAM",
})

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def shell_profile_env(timeout_s: float = 5.0) -> dict[str, str]:
  """Re-read user env vars from the zsh profile files (e.g. ~/.zshenv).

  ``/restart`` spawns the replacement Nemo with the *current* process env
  (subprocess inherits os.environ), so a key added or rotated in a shell
  profile after the daemon started never reaches the restarted daemon — e.g.
  "Preset oc-… needs $OPENCODE_GO_API_KEY in the daemon env" after a mid-run
  ``/restart``. Re-source the profile files in a non-interactive zsh and
  return exactly the profile-defined vars so the caller can layer them over
  the process env. Returns {} if zsh or the profile is unavailable, so a
  restart never blocks on it.
  """
  home = os.environ.get("HOME")
  if not home:
    return {}
  try:
    result = subprocess.run(
      ["/bin/zsh", "-c",
       '[[ -f "$HOME/.zprofile" ]] && source "$HOME/.zprofile" 2>/dev/null; env'],
      capture_output=True,
      text=True,
      timeout=timeout_s,
      check=False,
    )
  except (OSError, subprocess.TimeoutExpired) as exc:
    _LOG.warning("Skipping shell profile env on restart: %s", exc)
    return {}
  if result.returncode != 0:
    _LOG.warning(
      "Skipping shell profile env on restart (zsh rc=%d): %s",
      result.returncode, result.stderr.strip(),
    )
    return {}
  overlay: dict[str, str] = {}
  for line in result.stdout.splitlines():
    if "=" not in line:
      continue
    key, _, value = line.partition("=")
    if key in _PROTECTED_RESTART_ENV or not _ENV_KEY_RE.match(key):
      continue
    overlay[key] = value
  return overlay


def spawn_lifecycle_helper(
  spec: RestartSpec,
  *,
  upgrade: bool = False,
  original_args: list[str] | None = None,
) -> str:
  """Start a detached helper that optionally upgrades, then restarts Nemo."""
  parent_pid = os.getpid()
  restart_args = build_restart_args(spec, original_args)
  log_path = helper_log_path(parent_pid)
  helper_args = [
    sys.executable,
    "-m",
    "nemo.lifecycle",
    "--parent-pid",
    str(parent_pid),
    "--log-path",
    log_path,
  ]
  if upgrade:
    helper_args.append("--upgrade")
  helper_args.append("--")
  helper_args.extend(restart_args)
  # Layer the current shell profile's env over the daemon's env so the
  # replacement picks up keys added or rotated since the daemon started.
  env = dict(os.environ)
  env.update(shell_profile_env())
  with open(log_path, "ab", buffering=0) as log_file:
    subprocess.Popen(
      helper_args,
      stdin=subprocess.DEVNULL,
      stdout=log_file,
      stderr=log_file,
      start_new_session=True,
      close_fds=True,
      env=env,
    )
  return log_path


def run_pipx_upgrade(timeout_s: float = 300.0) -> CommandResult:
  """Run ``pipx upgrade captain-nemo`` and capture combined output."""
  try:
    result = subprocess.run(
      ["pipx", "upgrade", PACKAGE_NAME],
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      timeout=timeout_s,
    )
  except FileNotFoundError:
    return CommandResult(127, "`pipx` not found in PATH.")
  except subprocess.TimeoutExpired as exc:
    output = exc.stdout if isinstance(exc.stdout, str) else ""
    return CommandResult(124, output + "\nTimed out running `pipx upgrade captain-nemo`.")
  return CommandResult(result.returncode, result.stdout)


def _version_key(version: str) -> tuple[int, ...]:
  """Best-effort comparable key for Nemo's numeric release versions."""
  parts: list[int] = []
  for raw in version.replace("-", ".").split("."):
    digits = ""
    for ch in raw:
      if ch.isdigit():
        digits += ch
      else:
        break
    if digits:
      parts.append(int(digits))
    else:
      break
  return tuple(parts)


def check_pypi_upgrade(timeout_s: float = 10.0) -> UpgradeCheckResult:
  """Check PyPI for the latest captain-nemo version without installing."""
  from .version import get_version_info

  info = get_version_info()
  current = info.version
  url = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
  try:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
      payload: object = json.loads(resp.read())
  except urllib.error.URLError as exc:
    return UpgradeCheckResult(
      1, current, "", False, f"Failed to query PyPI: {exc}")
  except (TimeoutError, json.JSONDecodeError) as exc:
    return UpgradeCheckResult(
      1, current, "", False, f"Failed to parse PyPI response: {exc}")

  if not isinstance(payload, dict):
    return UpgradeCheckResult(1, current, "", False, "Unexpected PyPI response.")
  info_obj = payload.get("info")
  if not isinstance(info_obj, dict):
    return UpgradeCheckResult(1, current, "", False, "PyPI response missing info.")
  latest_obj = info_obj.get("version")
  if not isinstance(latest_obj, str) or not latest_obj:
    return UpgradeCheckResult(1, current, "", False, "PyPI response missing version.")

  latest = latest_obj
  latest_key = _version_key(latest)
  current_key = _version_key(current)
  update_available = latest_key > current_key
  if update_available:
    output = (
      f"Update available: current `{current}`, latest `{latest}`. "
      "Run `/upgrade` to install it with pipx and restart."
    )
  elif current_key > latest_key:
    output = (
      f"Running version `{current}` is newer than PyPI latest `{latest}`."
    )
  else:
    output = f"Nemo is up to date: current `{current}`, latest `{latest}`."
  if info.source == "source checkout":
    output += (
      "\n\nRunning from source checkout; `/upgrade` only applies to pipx "
      "installs. Update the checkout and use `/restart`."
    )
  return UpgradeCheckResult(0, current, latest, update_available, output)


def _pid_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except OSError:
    return False
  return True


def _wait_for_parent_exit(pid: int, timeout_s: float = 30.0) -> None:
  deadline = time.time() + timeout_s
  while time.time() < deadline:
    if not _pid_alive(pid):
      return
    time.sleep(0.25)


def _run_helper(parent_pid: int, log_path: str, restart_args: list[str], upgrade: bool) -> int:
  print(f"lifecycle helper started parent={parent_pid} upgrade={upgrade}", flush=True)
  if upgrade:
    print("running: pipx upgrade captain-nemo", flush=True)
    result = run_pipx_upgrade()
    print(result.output, end="", flush=True)
    if result.returncode != 0:
      print(f"upgrade failed with exit code {result.returncode}", flush=True)
      return result.returncode

  _wait_for_parent_exit(parent_pid)
  cmd = [sys.executable, "-m", "nemo", *restart_args]
  print(f"restarting: {cmd!r}", flush=True)
  with open(log_path, "ab", buffering=0) as log_file:
    subprocess.Popen(
      cmd,
      stdin=subprocess.DEVNULL,
      stdout=log_file,
      stderr=log_file,
      start_new_session=True,
      close_fds=True,
    )
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(prog="python -m nemo.lifecycle")
  parser.add_argument("--parent-pid", type=int, required=True)
  parser.add_argument("--log-path", required=True)
  parser.add_argument("--upgrade", action="store_true")
  parser.add_argument("restart_args", nargs=argparse.REMAINDER)
  args = parser.parse_args()
  restart_args = list(args.restart_args)
  if restart_args and restart_args[0] == "--":
    restart_args = restart_args[1:]
  return _run_helper(args.parent_pid, args.log_path, restart_args, args.upgrade)


if __name__ == "__main__":
  raise SystemExit(main())
