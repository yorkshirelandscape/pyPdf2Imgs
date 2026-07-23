#!/bin/bash
# Removes the macOS quarantine flag so PyPdf2Imgs.app opens without Gatekeeper
# flagging it. This is needed because the app isn't signed with a paid Apple
# Developer ID; it doesn't do anything to the app other than clear that flag.
DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$DIR/PyPdf2Imgs.app"

if [ ! -d "$APP" ]; then
    echo "Could not find PyPdf2Imgs.app next to this script."
    echo "Make sure this file stays in the same folder you unzipped it into."
    read -p "Press Enter to close this window..."
    exit 1
fi


# Target the quarantine attribute specifically rather than clearing everything
# (-cr): newer macOS versions also add a com.apple.provenance attribute that
# isn't ours to remove and can make a blanket clear fail outright, even
# though quarantine (the only thing that actually blocks launching) comes off
# fine on its own.
xattr -d -r com.apple.quarantine "$APP" 2>/dev/null
echo "Done. Try opening PyPdf2Imgs.app now -- it should launch without a warning."
read -p "Press Enter to close this window..."
