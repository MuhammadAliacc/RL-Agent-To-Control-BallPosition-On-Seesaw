@echo off
REM Navigate to your project directory
cd /d "C:\Users\muali1\Downloads\Muhammad_Ali\Dashboard"

REM Activate the virtual environment
call myenv\Scripts\activate

REM Run your Python script
python start_ball_beam_app.py

REM Keep the window open to show messages
pause
