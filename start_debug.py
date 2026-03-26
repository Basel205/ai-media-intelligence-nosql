"""
start_debug.py — Same as start.pyw but keeps the terminal open.
Use this during development to see logs and errors.

Run with:
    conda activate SIH
    python start_debug.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tray.tray_app import TrayApp

if __name__ == "__main__":
    app = TrayApp()
    app.start()
