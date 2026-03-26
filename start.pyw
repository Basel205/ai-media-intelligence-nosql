"""
start.pyw — Main entry point for the Media Intelligence system tray app.

The .pyw extension tells Windows to run this with pythonw.exe,
which suppresses the terminal window entirely.

To set up auto-start on Windows login:
1. Press Win+R, type: shell:startup
2. Create a shortcut to this file in that folder
3. In the shortcut properties, set:
   Target: C:\Users\kbase\anaconda3\envs\SIH\pythonw.exe start.pyw
   Start in: C:\Users\kbase\Projects\media-intelligence-nosql
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Redirect stdout/stderr to a log file (since there's no terminal)
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")
sys.stdout = open(log_path, "a", buffering=1)
sys.stderr = sys.stdout

from tray.tray_app import TrayApp

if __name__ == "__main__":
    app = TrayApp()
    app.start()
