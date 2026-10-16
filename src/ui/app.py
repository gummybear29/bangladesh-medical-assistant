"""Streamlit entry point alias for src/ui/app.py."""

from pathlib import Path
import runpy

target = Path(__file__).parent / "streamlit_app.py"
runpy.run_path(str(target), run_name="__main__")
