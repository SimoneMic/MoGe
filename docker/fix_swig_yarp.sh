#!/usr/bin/env bash
#
# Fix YARP python bindings build failure caused by SWIG 4.2.0 on Ubuntu 24.04.
# See https://github.com/robotology/yarp/issues/3083
#
# Replaces the buggy apt SWIG (4.2.0) with YARP's prebuilt SWIG 4.3.0, then
# reconfigures the YARP build so CMake picks up the new binary.
#
# Run with: bash fix_yarp_swig.sh
# (uses sudo for the system-wide SWIG replacement)

set -euo pipefail

SWIG_ZIP_URL="https://github.com/robotology/robotology-vcpkg-ports/releases/download/storage/swig_4_3_0_ubuntu_24_04.zip"
YARP_BUILD_DIR="/home/user1/yarp/build"
WORK_DIR="$(mktemp -d)"

cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

echo "==> Current SWIG version (before):"
swig -version 2>/dev/null | grep -i version || echo "  (no swig found)"

echo "==> Removing buggy apt SWIG (4.2.0)..."
sudo apt-get purge -y swig swig4.0 || true

echo "==> Downloading SWIG 4.3.0..."
cd "$WORK_DIR"
wget -nv "$SWIG_ZIP_URL"
unzip -q swig_4_3_0_ubuntu_24_04.zip

echo "==> Installing SWIG 4.3.0 into /usr/bin and /usr/share..."
sudo mv swig_4_3_0_ubuntu_24_04_install/bin/* /usr/bin/
sudo mv swig_4_3_0_ubuntu_24_04_install/share/swig /usr/share/

echo "==> SWIG version (after):"
swig -version | grep -i version

# Sanity check: must be 4.3.0
if ! swig -version | grep -q "4.3.0"; then
    echo "ERROR: SWIG is not 4.3.0 after install. Aborting before touching the build." >&2
    exit 1
fi

#echo "==> Reconfiguring YARP build to use the new SWIG..."
#if [ -d "$YARP_BUILD_DIR" ]; then
#    cmake -DSWIG_EXECUTABLE=/usr/bin/swig -DSWIG_DIR=/usr/share/swig/4.3.0 "$YARP_BUILD_DIR"
#    echo "==> Done. Now build with:  cmake --build $YARP_BUILD_DIR -j\$(nproc)"
#else
#    echo "WARNING: $YARP_BUILD_DIR not found. Configure/build YARP manually;"
#    echo "         the SWIG 4.3.0 install above is what matters."
#fi
