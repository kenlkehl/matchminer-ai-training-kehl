#!/usr/bin/env bash
set -euo pipefail

LLAMA_REPO_URL="https://github.com/ggml-org/llama.cpp"

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${LLAMA_CPP_DIR:-/tmp/llama.cpp}"
LLAMA_REF="${LLAMA_CPP_REF:-master}"
ARCH="${MATCHMINER_MAC_ARCH:-}"
RUN_NPM_CI=1
BUILD_LLAMA=1
BUILD_ELECTRON=1
SMOKE_TEST=1
ADHOC_SIGN=1
UNSIGNED=0

usage() {
  cat <<'USAGE'
Usage: ./build_for_mac.sh [options]

Builds a macOS Electron DMG/zip for MatchMiner AI and bundles a local
llama.cpp llama-server binary into resources/bin.

Options:
  --arch arm64|x64          Target Mac architecture. Defaults to this machine.
  --llama-ref REF           llama.cpp branch/tag/commit. Defaults to master.
  --llama-dir DIR           llama.cpp checkout directory. Defaults to /tmp/llama.cpp.
  --skip-npm-ci             Reuse the existing node_modules.
  --skip-llama              Reuse resources/bin/llama-server.
  --skip-electron-dist      Build/stage llama.cpp only.
  --skip-smoke-test         Do not run llama-server --help after staging.
  --no-ad-hoc-sign          Do not ad-hoc sign staged llama.cpp files.
  --unsigned                Disable electron-builder certificate auto-discovery.
  -h, --help                Show this help.

Environment overrides:
  MATCHMINER_MAC_ARCH       Same as --arch.
  LLAMA_CPP_REF             Same as --llama-ref.
  LLAMA_CPP_DIR             Same as --llama-dir.
  LLAMA_CPP_BUILD_DIR       Override the llama.cpp CMake build directory.
  JOBS                      Parallel build jobs. Defaults to sysctl hw.ncpu.

Examples:
  ./build_for_mac.sh
  ./build_for_mac.sh --arch arm64
  ./build_for_mac.sh --arch x64 --unsigned
USAGE
}

log() {
  printf '\n==> %s\n' "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      ARCH="${2:-}"
      [[ -n "$ARCH" ]] || die "--arch requires arm64 or x64"
      shift 2
      ;;
    --llama-ref)
      LLAMA_REF="${2:-}"
      [[ -n "$LLAMA_REF" ]] || die "--llama-ref requires a branch, tag, or commit"
      shift 2
      ;;
    --llama-dir)
      LLAMA_DIR="${2:-}"
      [[ -n "$LLAMA_DIR" ]] || die "--llama-dir requires a path"
      shift 2
      ;;
    --skip-npm-ci)
      RUN_NPM_CI=0
      shift
      ;;
    --skip-llama)
      BUILD_LLAMA=0
      shift
      ;;
    --skip-electron-dist)
      BUILD_ELECTRON=0
      shift
      ;;
    --skip-smoke-test)
      SMOKE_TEST=0
      shift
      ;;
    --no-ad-hoc-sign)
      ADHOC_SIGN=0
      shift
      ;;
    --unsigned)
      UNSIGNED=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || die "macOS builds must be run on macOS."

case "${ARCH:-$(uname -m)}" in
  arm64)
    ARCH="arm64"
    CMAKE_OSX_ARCH="arm64"
    ;;
  x64|x86_64|amd64)
    ARCH="x64"
    CMAKE_OSX_ARCH="x86_64"
    ;;
  *)
    die "Unsupported --arch value '$ARCH'. Use arm64 or x64."
    ;;
esac

JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
LLAMA_BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-${LLAMA_DIR}/build-macos-${ARCH}}"
RESOURCE_BIN_DIR="${APP_DIR}/resources/bin"
STAGED_LLAMA_SERVER="${RESOURCE_BIN_DIR}/llama-server"

verify_cpp_toolchain() {
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  printf '#include <mutex>\n#include <cstdio>\nint main(){ std::mutex m; return 0; }\n' > "${tmp_dir}/cxxcheck.cpp"
  if ! /usr/bin/c++ -std=c++17 "${tmp_dir}/cxxcheck.cpp" -o "${tmp_dir}/cxxcheck" >"${tmp_dir}/stdout" 2>"${tmp_dir}/stderr"; then
    printf '\nAppleClang cannot find the standard C++ library headers.\n' >&2
    printf 'Fix Command Line Tools/Xcode, then rerun this script:\n\n' >&2
    printf '  sudo rm -rf /Library/Developer/CommandLineTools\n' >&2
    printf '  xcode-select --install\n' >&2
    printf '  sudo xcodebuild -license accept\n\n' >&2
    cat "${tmp_dir}/stderr" >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi
  rm -rf "${tmp_dir}"
}

prepare_llama_source() {
  if [[ -d "${LLAMA_DIR}/.git" ]]; then
    log "Updating llama.cpp checkout at ${LLAMA_DIR}"
    git -C "${LLAMA_DIR}" fetch --tags origin
  else
    log "Cloning llama.cpp into ${LLAMA_DIR}"
    mkdir -p "$(dirname "${LLAMA_DIR}")"
    git clone "${LLAMA_REPO_URL}" "${LLAMA_DIR}"
  fi

  git -C "${LLAMA_DIR}" checkout "${LLAMA_REF}"
  if [[ "${LLAMA_REF}" == "master" || "${LLAMA_REF}" == "main" ]]; then
    git -C "${LLAMA_DIR}" pull --ff-only origin "${LLAMA_REF}"
  fi
}

build_llama_server() {
  log "Building llama-server for macOS ${ARCH}"
  rm -rf "${LLAMA_BUILD_DIR}"
  cmake -S "${LLAMA_DIR}" -B "${LLAMA_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES="${CMAKE_OSX_ARCH}" \
    -DGGML_METAL=ON \
    -DGGML_CCACHE=OFF
  cmake --build "${LLAMA_BUILD_DIR}" --target llama-server -j"${JOBS}"
}

find_built_llama_server() {
  local candidate="${LLAMA_BUILD_DIR}/bin/llama-server"
  if [[ -x "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return
  fi
  candidate="$(find "${LLAMA_BUILD_DIR}" -type f -name llama-server -perm -111 -print -quit 2>/dev/null || true)"
  [[ -n "${candidate}" ]] || die "Could not find built llama-server under ${LLAMA_BUILD_DIR}"
  printf '%s\n' "${candidate}"
}

stage_llama_files() {
  local llama_server
  llama_server="$(find_built_llama_server)"

  log "Staging llama.cpp files into ${RESOURCE_BIN_DIR}"
  mkdir -p "${RESOURCE_BIN_DIR}"
  cp "${llama_server}" "${STAGED_LLAMA_SERVER}"
  chmod 755 "${STAGED_LLAMA_SERVER}"

  if [[ -d "${LLAMA_BUILD_DIR}/bin" ]]; then
    find "${LLAMA_BUILD_DIR}/bin" -maxdepth 1 -type f \( -name '*.dylib' -o -name '*.metallib' \) -print0 |
      while IFS= read -r -d '' file; do
        cp "${file}" "${RESOURCE_BIN_DIR}/"
        chmod 755 "${RESOURCE_BIN_DIR}/$(basename "${file}")" || true
      done
  fi
}

is_system_dependency() {
  case "$1" in
    /System/Library/*|/usr/lib/*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

rewrite_dependency_to_local() {
  local binary="$1"
  local dependency="$2"
  local basename_dependency
  basename_dependency="$(basename "${dependency}")"

  [[ -f "${RESOURCE_BIN_DIR}/${basename_dependency}" ]] || return 0
  install_name_tool -change "${dependency}" "@executable_path/${basename_dependency}" "${binary}" 2>/dev/null || true
}

bundle_non_system_dylibs() {
  log "Bundling non-system dynamic libraries needed by llama-server"

  local changed=1
  while [[ "${changed}" -eq 1 ]]; do
    changed=0
    while IFS= read -r -d '' binary; do
      while IFS= read -r dependency; do
        [[ -n "${dependency}" ]] || continue
        is_system_dependency "${dependency}" && continue

        local basename_dependency
        basename_dependency="$(basename "${dependency}")"

        case "${dependency}" in
          @executable_path/*)
            ;;
          @rpath/*|@loader_path/*)
            rewrite_dependency_to_local "${binary}" "${dependency}"
            ;;
          /*)
            if [[ -f "${dependency}" ]]; then
              if [[ ! -f "${RESOURCE_BIN_DIR}/${basename_dependency}" ]]; then
                cp "${dependency}" "${RESOURCE_BIN_DIR}/"
                chmod 755 "${RESOURCE_BIN_DIR}/${basename_dependency}" || true
                changed=1
              fi
              rewrite_dependency_to_local "${binary}" "${dependency}"
            fi
            ;;
        esac
      done < <(otool -L "${binary}" | tail -n +2 | awk '{print $1}')
    done < <(find "${RESOURCE_BIN_DIR}" -maxdepth 1 -type f \( -name llama-server -o -name '*.dylib' \) -print0)
  done

  find "${RESOURCE_BIN_DIR}" -maxdepth 1 -type f -name '*.dylib' -print0 |
    while IFS= read -r -d '' dylib; do
      install_name_tool -id "@executable_path/$(basename "${dylib}")" "${dylib}" 2>/dev/null || true
    done
}

ad_hoc_sign_staged_files() {
  [[ "${ADHOC_SIGN}" -eq 1 ]] || return 0
  require_command codesign

  log "Ad-hoc signing staged llama.cpp files"
  find "${RESOURCE_BIN_DIR}" -maxdepth 1 -type f \( -name llama-server -o -name '*.dylib' \) -print0 |
    while IFS= read -r -d '' file; do
      codesign --force --sign - "${file}" >/dev/null
    done
}

smoke_test_llama_server() {
  [[ "${SMOKE_TEST}" -eq 1 ]] || return 0
  log "Smoke-testing staged llama-server"
  "${STAGED_LLAMA_SERVER}" --help >/dev/null
}

print_dynamic_libraries() {
  log "llama-server dynamic libraries"
  otool -L "${STAGED_LLAMA_SERVER}"
}

build_electron_app() {
  if [[ "${UNSIGNED}" -eq 1 ]]; then
    export CSC_IDENTITY_AUTO_DISCOVERY=false
  fi

  log "Building Electron macOS ${ARCH} DMG and zip"
  npm run electron:dist -- --mac dmg zip "--${ARCH}"
}

print_outputs() {
  log "Build outputs"
  find "${APP_DIR}/release" -maxdepth 1 -type f \( -name '*.dmg' -o -name '*.zip' \) -print | sort || true
}

require_command git
require_command cmake
require_command npm
require_command otool
require_command install_name_tool
require_command xcrun
verify_cpp_toolchain

cd "${APP_DIR}"

if [[ "${RUN_NPM_CI}" -eq 1 ]]; then
  log "Installing npm dependencies"
  npm ci
fi

if [[ "${BUILD_LLAMA}" -eq 1 ]]; then
  prepare_llama_source
  build_llama_server
  stage_llama_files
  bundle_non_system_dylibs
  ad_hoc_sign_staged_files
  smoke_test_llama_server
  print_dynamic_libraries
else
  [[ -x "${STAGED_LLAMA_SERVER}" ]] || die "--skip-llama was set, but ${STAGED_LLAMA_SERVER} is missing or not executable."
fi

if [[ "${BUILD_ELECTRON}" -eq 1 ]]; then
  build_electron_app
  print_outputs
fi
