# PyPdf2Imgs

A modern, user-friendly tool for extracting and saving images from PDF files using a graphical interface. Built with Python and PyQt6, it leverages [mutool](https://mupdf.com/docs/manual-mutool.html) for fast, accurate image extraction.

## Features
- Extracts all images from a PDF file with a single click
- Displays thumbnails in a scrollable, selectable grid
- Batch select/deselect images
- Rename images before saving
- Save selected images as PNG or WEBP

## Requirements
- **mutool** (from MuPDF)
  - Download and install from: https://mupdf.com/downloads/
  - Ensure `mutool` is in your system PATH

## Installation
1. Download the appropriate executable from GitHub Releases:
   - [macOS](https://github.com/yorkshirelandscape/pyPdf2Imgs/releases/latest/download/PyPdf2Imgs-macos.zip)
   - [Windows](https://github.com/yorkshirelandscape/pyPdf2Imgs/releases/latest/download/PyPdf2Imgs-windows.zip)
   - [Linux](https://github.com/yorkshirelandscape/pyPdf2Imgs/releases/latest/download/PyPdf2Imgs-linux.zip)
2. Install [mutool](https://mupdf.com/downloads/) for your platform and ensure it is available in your PATH:
   - **macOS:** `brew install mupdf-tools`
   - **Windows/Linux:** download from [mupdf.com](https://mupdf.com/downloads/), or use your package manager (e.g. `sudo apt install mupdf-tools`)

## Usage
1. Run the application.
2. Click "Open PDF" (either the button in the center of the window, or the toolbar button) and select a PDF file.
3. Browse, select, and rename images as desired.
4. Click "Save Selected" to export images to your chosen folder.

## Run from source or build your own executable

### Requirements
- [Python 3.9+](https://www.python.org/downloads/)
- Python packages:
  - `Pillow`
  - `PyQt6`
  - `pyinstaller` (only needed to build your own executable)

### Run from source
1. Install pipenv: `pip install --user pipenv`
2. Clone the repository: `git clone https://github.com/yorkshirelandscape/pyPdf2Imgs.git`
3. Navigate into the directory: `cd pyPdf2Imgs`
4. Install dependencies: `pipenv install`
5. Run it: `pipenv run python main.py`

### Build your own executable
1. Install dev dependencies: `pipenv install --dev`
2. Build with PyInstaller, bundling the spinner asset:
   - **macOS/Linux:** `pipenv run pyinstaller --onefile --windowed --name PyPdf2Imgs --add-data "spinner.gif:." main.py`
   - **Windows:** `pipenv run pyinstaller --onefile --windowed --name PyPdf2Imgs --add-data "spinner.gif;." main.py`
3. Find the executable in `dist/`.

## License
[MIT License](LICENSE)
