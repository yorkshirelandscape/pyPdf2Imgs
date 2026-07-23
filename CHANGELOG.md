# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
