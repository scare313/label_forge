@echo off
REM Change directory to your project folder
cd /d "%~dp0"

REM Activate virtual environment
call auto_fba_label\Scripts\activate

REM Run Streamlit app
streamlit run sl_make_labels_v1.py

REM Keep window open after execution
pause