@echo off
title Skill Demand Structure Discovery Dashboard - Group G25
cd /d "%~dp0"
set USERPROFILE=%~dp0
set HOME=%~dp0
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
echo =======================================================================
echo Launching Skill Demand Structure Discovery Dashboard
echo Group: G25 | Student: Jashandeep Singh Gill (12504818)
echo Python 3.15.2 ^| Streamlit 1.67 ^| Scikit-Learn 1.9
echo =======================================================================
python run_dashboard.py
pause
