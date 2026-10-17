"""Streamlit Community Cloud Entry Point for Bangladesh Medical & First Aid Assistant."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Run the full-featured Streamlit UI
from src.ui.streamlit_app import *
