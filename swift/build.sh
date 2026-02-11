#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/AudioCaptureHelper"
echo "Building AudioCaptureHelper..." >&2
swift build -c release 2>&1 | tail -3 >&2
# Copy binary up for easy access
cp .build/release/AudioCaptureHelper "$SCRIPT_DIR/.build/release/AudioCaptureHelper" 2>/dev/null || {
    mkdir -p "$SCRIPT_DIR/.build/release"
    cp .build/release/AudioCaptureHelper "$SCRIPT_DIR/.build/release/AudioCaptureHelper"
}
echo "Built: $SCRIPT_DIR/.build/release/AudioCaptureHelper" >&2
