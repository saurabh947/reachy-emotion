"""System dependency management: check and install ffmpeg and portaudio.

These are C-library dependencies that pip cannot install — they must be
provided by the OS package manager (apt on Linux, brew on macOS).

Usage
─────
At app startup (warn only):
    from reachy_emotion.system_deps import check_and_warn
    check_and_warn()

From the install script or CLI:
    from reachy_emotion.system_deps import install_missing
    install_missing()
"""

import logging
import platform
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency definitions
# ---------------------------------------------------------------------------

_DEPS: dict[str, dict] = {
    "ffmpeg": {
        "check": lambda: shutil.which("ffmpeg") is not None,
        "apt": ["ffmpeg"],
        "brew": ["ffmpeg"],
        "winget": ["Gyan.FFmpeg"],
        "purpose": "TTS audio conversion (MP3 → WAV for Reachy speaker)",
        "manual": "https://ffmpeg.org/download.html",
    },
    "portaudio": {
        "check": _check_portaudio,
        "apt": ["libportaudio2", "portaudio19-dev"],
        "brew": ["portaudio"],
        "winget": None,  # bundled in SpeechRecognition wheel on Windows
        "purpose": "Microphone access (required by SpeechRecognition)",
        "manual": "http://www.portaudio.com/download.html",
    },
}


def _check_portaudio() -> bool:
    """Return True if portaudio is usable (sounddevice or pyaudio can load it)."""
    try:
        import sounddevice  # noqa: F401
        return True
    except (ImportError, OSError):
        pass
    try:
        import pyaudio  # noqa: F401
        return True
    except (ImportError, OSError):
        pass
    return False


# Patch the forward-referenced lambda after defining the function
_DEPS["portaudio"]["check"] = _check_portaudio


# ---------------------------------------------------------------------------
# OS detection helpers
# ---------------------------------------------------------------------------

def _os() -> str:
    """Return 'linux', 'macos', or 'windows'."""
    s = platform.system().lower()
    if s == "darwin":
        return "macos"
    if s == "windows":
        return "windows"
    return "linux"


def _has_brew() -> bool:
    return shutil.which("brew") is not None


def _has_apt() -> bool:
    return shutil.which("apt-get") is not None


def _has_winget() -> bool:
    return shutil.which("winget") is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def missing() -> list[str]:
    """Return names of system dependencies that are not currently available."""
    return [name for name, info in _DEPS.items() if not info["check"]()]


def check_and_warn() -> bool:
    """Log a clear warning for each missing system dependency. Returns True if all present.

    Call this at app startup so users see actionable messages rather than
    obscure import errors later.
    """
    absent = missing()
    if not absent:
        return True

    os_name = _os()
    for name in absent:
        info = _DEPS[name]
        hint = _install_hint(name, os_name)
        logger.warning(
            "%s not found — %s will be degraded. Install with: %s",
            name,
            info["purpose"],
            hint,
        )
    return False


def install_missing(*, dry_run: bool = False) -> bool:
    """Attempt to install all missing system dependencies via the OS package manager.

    Args:
        dry_run: If True, print what would be run without executing.

    Returns:
        True if all dependencies are present after the operation.
    """
    absent = missing()
    if not absent:
        print("✅ All system dependencies already installed.")
        return True

    os_name = _os()

    for name in absent:
        info = _DEPS[name]
        print(f"→ Installing {name} ({info['purpose']}) …")
        success = _run_install(name, os_name, dry_run=dry_run)
        if not success:
            hint = _install_hint(name, os_name)
            print(f"  ⚠️  Could not install {name} automatically. Install manually: {hint}")

    # Re-check after install attempts
    still_missing = missing()
    if still_missing:
        print(f"\n⚠️  Still missing: {', '.join(still_missing)}")
        print("   Some features will be disabled until these are installed.")
        return False

    print("✅ All system dependencies installed.")
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _install_hint(name: str, os_name: str) -> str:
    info = _DEPS[name]
    if os_name == "linux" and _has_apt():
        pkgs = " ".join(info["apt"])
        return f"sudo apt-get install -y {pkgs}"
    if os_name == "macos" and _has_brew():
        pkgs = " ".join(info["brew"])
        return f"brew install {pkgs}"
    if os_name == "windows" and info.get("winget") and _has_winget():
        pkg = info["winget"]
        return f"winget install {pkg}"
    return info["manual"]


def _run_install(name: str, os_name: str, *, dry_run: bool) -> bool:
    info = _DEPS[name]
    cmd: list[str] | None = None

    if os_name == "linux" and _has_apt():
        cmd = ["sudo", "apt-get", "install", "-y"] + info["apt"]
    elif os_name == "macos" and _has_brew():
        cmd = ["brew", "install"] + info["brew"]
    elif os_name == "windows" and info.get("winget") and _has_winget():
        cmd = ["winget", "install", "--silent", info["winget"]]

    if cmd is None:
        return False

    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return True

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        logger.debug("Install command failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CLI entry point  (`reachy-emotion-setup` script)
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """Check and install system dependencies. Run once after `pip install -e .`."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check and install system dependencies for reachy-emotion"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be installed without running any commands"
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Report missing dependencies without attempting to install"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    absent = missing()
    if not absent:
        print("✅ All system dependencies satisfied:")
        for name in _DEPS:
            print(f"   • {name}")
        sys.exit(0)

    print(f"⚠️  Missing system dependencies: {', '.join(absent)}\n")

    if args.check_only:
        for name in absent:
            print(f"   • {name}: {_install_hint(name, _os())}")
        sys.exit(1)

    ok = install_missing(dry_run=args.dry_run)
    sys.exit(0 if ok else 1)
