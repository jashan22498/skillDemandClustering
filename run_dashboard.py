"""
Dashboard Launcher for Skill Demand Structure Discovery Platform
Author: Jashandeep Singh Gill (12504818) - Group G25
Compatible with Python 3.15.2
"""

import os
import sys
import subprocess

def launch():
    workspace = os.path.abspath(os.path.dirname(__file__))
    os.chdir(workspace)

    # Ensure environment points to local .streamlit directory
    os.environ["USERPROFILE"] = workspace
    os.environ["HOME"] = workspace
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false"
    ]

    print("=======================================================================")
    print("🚀 Launching Skill Demand Structure Discovery Dashboard")
    print("   Author: Jashandeep Singh Gill (12504818) | Group G25")
    print("   Platform: Python 3.15.2 | Streamlit 1.67 | Scikit-Learn 1.9")
    print("=======================================================================")
    print("Opening in your web browser at http://localhost:8501 ...")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    launch()
