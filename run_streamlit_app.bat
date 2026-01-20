@echo off

REM venvが存在しない場合は作成
if not exist venv (
    echo Creating virtual environment...
    py -3.11 -m venv venv
) else (
    echo Virtual environment already exists.
)

REM venvを有効化
call venv\Scripts\activate.bat

REM 依存関係をインストール
pip install -r requirements.txt

REM Streamlitを起動
streamlit run apps/streamlit_app.py

pause
