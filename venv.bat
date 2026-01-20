@echo off
REM venv‚ğì¬
py -3.11 -m venv venv

REM venv‚ğ—LŒø‰»
call venv\Scripts\activate.bat

pip install -r requirements.txt

streamlit run apps/streamlit_app.py

pause
