#!/bin/sh
# Capture a session of stills from a Microduck's own camera. Runs ON THE ROBOT, pushed over ssh
# by `duck_detector.capture` — nothing to install on the board.
#
# Every setting arrives in the environment rather than as an argument, because this script is fed
# to `sh -s` on stdin and there is no argv worth threading through ssh quoting:
#
#   DIR      where to write the frames (created)
#   SECONDS_ how long to capture, wall clock
#   HZ_NUM   frames per second to keep, as a fraction — GStreamer caps have no floats, and
#   HZ_DEN   `framerate=2.0/1` is a negotiation failure, not a rounding
#   WIDTH    capture geometry, before the rotation
#   HEIGHT
#   FLIP     videoflip video-direction: `90r`, `180`, `90l` or `identity`
#   DEVICE   capture node, /dev/video0
#   QUALITY  JPEG quality
#
# **`mediad` holds the camera**, and V4L2 capture is exclusive: this stops it, captures, and
# starts it again if it was running — through a trap, so a failure or a Ctrl-C on the far end
# still gives the robot its camera daemon back.
set -e

DIR="${DIR:?DIR}"
SECONDS_="${SECONDS_:-60}"
HZ_NUM="${HZ_NUM:-2}"
HZ_DEN="${HZ_DEN:-1}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FLIP="${FLIP:-90r}"
DEVICE="${DEVICE:-/dev/video0}"
QUALITY="${QUALITY:-90}"

say() { printf '== %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v gst-launch-1.0 >/dev/null 2>&1 \
    || die "no gst-launch-1.0 on this board. sudo /usr/local/sbin/robot-setup-gstreamer"

# `sudo` needs a terminal here, and the caller gives it one (`ssh -t`, as the robot's own scripts
# do) — so a password prompt is a prompt rather than a hang. Said out loud all the same, because a
# prompt appearing in the middle of a capture is a surprise otherwise.
if ! sudo -n true 2>/dev/null; then
    say "sudo will ask for your password (add a NOPASSWD rule for systemctl to capture unattended)"
fi
[ -e "$DEVICE" ] || die "$DEVICE does not exist — is the camera overlay active?"

MEDIAD_WAS=""
if systemctl is-active --quiet mediad 2>/dev/null; then
    MEDIAD_WAS=1
fi

restore() {
    if [ -n "$MEDIAD_WAS" ]; then
        say "starting mediad again"
        sudo systemctl start mediad || say "WARNING: could not start mediad"
    fi
}
trap restore EXIT INT TERM

if [ -n "$MEDIAD_WAS" ]; then
    say "stopping mediad — V4L2 capture is exclusive and it holds $DEVICE"
    sudo systemctl stop mediad
    # The device does not free instantly on a pipeline teardown.
    i=0
    while [ "$i" -lt 20 ]; do
        fuser "$DEVICE" >/dev/null 2>&1 || break
        i=$((i + 1))
        sleep 0.25
    done
fi

# The sensor's own mode, which `mediad` normally pins and the 3A engine reads once at startup.
# The helper is what `scripts/setup-rkaiq.sh` installs; without it, capture still works from the
# boot mode, only slower.
if [ -x /usr/local/bin/rkaiq-pin-sensor-mode ]; then
    sudo /usr/local/bin/rkaiq-pin-sensor-mode >/dev/null 2>&1 || true
fi

mkdir -p "$DIR"
rm -f "$DIR"/frame_*.jpg

# 30 fps off the sensor, decimated to HZ *before* the colour conversion so only the frames that
# are kept cost anything. The rotation is not optional: `mediad` turns the picture a quarter turn
# before its tee, so upright is what a model will see at inference — a dataset captured sideways
# trains a detector for a camera nobody has.
BUFFERS=$((30 * SECONDS_))
say "capturing ${SECONDS_}s at ${HZ_NUM}/${HZ_DEN} Hz (${WIDTH}x${HEIGHT}, flip ${FLIP}) into ${DIR}"
gst-launch-1.0 -e --no-position \
    v4l2src device="$DEVICE" num-buffers="$BUFFERS" \
    ! video/x-raw,format=UYVY,width="$WIDTH",height="$HEIGHT",framerate=30/1 \
    ! videorate ! video/x-raw,framerate="$HZ_NUM"/"$HZ_DEN" \
    ! videoconvert \
    ! videoflip video-direction="$FLIP" \
    ! jpegenc quality="$QUALITY" \
    ! multifilesink location="$DIR/frame_%05d.jpg" \
    >&2

COUNT=$(find "$DIR" -name 'frame_*.jpg' | wc -l)
say "wrote ${COUNT} frames"
[ "$COUNT" -gt 0 ] || die "no frames — is anything else holding $DEVICE? journalctl -u mediad -b"

# The one line stdout carries: the caller reads it and puts it in the session record.
printf '%s\n' "$COUNT"
