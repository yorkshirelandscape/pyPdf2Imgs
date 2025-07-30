# PyPdf2Imgs

A modern, user-friendly tool for extracting and saving images from PDF files using a graphical interface. Built with Python and Tkinter, it leverages [mutool](https://mupdf.com/docs/manual-mutool.html) for fast, accurate image extraction.

## Features
- Extracts all images from a PDF file with a single click
- Displays thumbnails in a scrollable, selectable grid
- Batch select/deselect images
- Rename images before saving
- Save selected images as PNG or WEBP

## Roadmap
- Responsive UI with animated spinner during processing
- Cross-platform mouse wheel/trackpad scrolling

## Requirements
- **Python 3.7+**
- **mutool** (from MuPDF)
  - Download and install from: https://mupdf.com/downloads/
  - Ensure `mutool` is in your system PATH
- Python packages: `Pillow`

## Installation
1. Install [mutool](https://mupdf.com/downloads/) for your platform and ensure it is available in your PATH.
2. Install Python dependencies:
   ```sh
   pip install Pillow
   ```
   Or use the provided `Pipfile`:
   ```sh
   pip install pipenv
   pipenv install
   ```
   *If using pipenv and vscode, I recommend setting the VENV_IN_PROJECT environment variable to true and telling vscode to use the pipenv environment, otherwise there's*
   ```sh
   pipenv run python main.py
   ```


## Usage
1. Run the application:
   ```sh
   python main.py
   ```
2. Select a PDF file when prompted.
3. Browse, select, and rename images as desired.
4. Click "Save Selected" to export images to your chosen folder.

## License
MIT License
