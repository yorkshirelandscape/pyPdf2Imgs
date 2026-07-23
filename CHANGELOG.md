# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-07-23
### Fixed
- The packaged macOS app crashed on launch (segfault in Qt's library path resolution). The release zip step was dereferencing the .app bundle's internal symlinks into full duplicate file copies, corrupting its structure; the zip step now preserves symlinks correctly

## [1.0.1] - 2026-07-23
### Fixed
- The packaged macOS app couldn't find mutool when launched normally (e.g. by double-clicking), since GUI apps don't inherit the shell's PATH; extraction now checks common install locations directly and reports a clear error if mutool truly isn't installed
- The macOS release zip contained a redundant duplicate of the app (a loose Unix executable alongside the .app bundle); only the .app is included now

### Changed
- Migrated the macOS build to PyInstaller's onedir mode; --onefile combined with a --windowed .app bundle is deprecated on macOS and will become a hard error in a future PyInstaller release
- Updated GitHub Actions dependencies (checkout, setup-python, upload-artifact, download-artifact) off the deprecated Node 20 runtime
- Documented the macOS Gatekeeper warning and the xattr -cr workaround in the README

## [1.0.0] - 2026-07-23
### Added
- PyQt6-based GUI, replacing the original Tkinter implementation
- Centered "Open PDF" button as the initial view, instead of auto-prompting a file dialog on launch
- Image selection shown via card border/background highlighting; click a thumbnail to toggle selection
- Per-image zoom, rename, and rotate actions
- Adaptive light/dark theming for the image grid that tracks the OS appearance live

### Fixed
- Background/full-page images that mutool extracts as JPEG were silently skipped, since the app assumed every extracted image was a PNG; extraction now resolves the actual file mutool produced for each image object
- Alpha-mask merging no longer fails when the base image is a JPEG (previously errored trying to save an RGBA result with a .jpg extension)
- Renaming an image now actually affects the exported filename (previously ignored by the save step)
