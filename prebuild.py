#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path
import re


def run(cmd, cwd=None):
    print(f"[run] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[error] Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        sys.exit(e.returncode)


def ensure_tool(name):
    if shutil.which(name) is None:
        print(f"[error] Required tool '{name}' not found in PATH.")
        sys.exit(1)


def cmake_version_tuple():
    try:
        out = subprocess.check_output(["cmake", "--version"], text=True)
        # first line like: "cmake version 3.16.3"
        m = re.search(r"version\s+(\d+)\.(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
    except Exception:
        pass
    return (0, 0, 0)


def build_whisper_cpp(cuda=True, cublas=False, build_type="Release"):
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "whisper.cpp"
    build_dir = src_dir / "build"

    if not src_dir.exists():
        print(f"[error] whisper.cpp source directory not found at {src_dir}")
        sys.exit(1)

    ensure_tool("cmake")

    # Check cmake version for CUDA builds (ggml-cuda requires >= 3.18)
    vmaj, vmin, vpatch = cmake_version_tuple()
    if cuda and (vmaj, vmin) < (3, 18):
        print(f"[warn] Detected CMake {vmaj}.{vmin}.{vpatch} (< 3.18). ggml-cuda requires >= 3.18.")
        print("[info] Falling back to CPU-only build (no CUDA). To enable CUDA, upgrade CMake.")
        print("       Quick upgrade options: 'sudo snap install cmake --classic' or Kitware APT repo.")
        cuda = False

    build_dir.mkdir(parents=True, exist_ok=True)

    ggml_cuda = "1" if cuda else "0"
    ggml_cublas = "1" if cublas else "0"

    print("[info] Generating build files with CMake...")
    run([
        "cmake",
        "-S", str(src_dir),
        "-B", str(build_dir),
        f"-DGGML_CUDA={ggml_cuda}",
        f"-DGGML_CUBLAS={ggml_cublas}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ])

    print("[info] Building whisper.cpp (this may take a few minutes)...")
    # Use parallel build according to CPU count when available
    jobs = str(os.cpu_count() or 4)
    run(["cmake", "--build", str(build_dir), "-j", jobs, "--config", build_type])

    # Verify outputs
    bin_path = build_dir / "bin" / "whisper-cli"
    lib_whisper = build_dir / "src" / "libwhisper.so"
    ggml_lib_dir = build_dir / "ggml" / "src"

    if not bin_path.exists():
        print(f"[error] Build succeeded but 'whisper-cli' not found at {bin_path}")
        sys.exit(1)

    # Make the binary executable
    try:
        os.chmod(bin_path, 0o755)
    except Exception as e:
        print(f"[warn] Could not set executable bit on {bin_path}: {e}")

    print("[ok] whisper.cpp build complete:")
    print(f"     - binary: {bin_path}")
    if lib_whisper.exists():
        print(f"     - libwhisper: {lib_whisper}")
    else:
        print("     - libwhisper: not found (CPU builds may place libs differently)")
    print(f"     - ggml libs directory: {ggml_lib_dir}")

    # Optional: quick sanity check
    try:
        subprocess.run([str(bin_path), "-h"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[ok] whisper-cli responds to -h")
    except Exception:
        print("[warn] whisper-cli sanity check failed; continuing")


def main():
    parser = argparse.ArgumentParser(description="Prebuild whisper.cpp artifacts for Docker prebuilt target")
    parser.add_argument("--cuda", action="store_true", default=True, help="Enable GGML_CUDA (default: on)")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="Disable GGML_CUDA")
    parser.add_argument("--cublas", action="store_true", default=False, help="Enable GGML_CUBLAS (default: off)")
    parser.add_argument("--build-type", default="Release", choices=["Release", "RelWithDebInfo", "Debug"], help="CMake build type")
    args = parser.parse_args()

    print("[start] Prebuilding whisper.cpp")
    build_whisper_cpp(cuda=args.cuda, cublas=args.cublas, build_type=args.build_type)
    print("[done] You can now run: 'docker compose build --no-cache' or the raw docker build command.")


if __name__ == "__main__":
    main()