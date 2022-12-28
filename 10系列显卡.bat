@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS= --xformers --lowvram --precision full --no-half  --opt-split-attention  --skip-torch-cuda-test

@REM --medvram
@REM --medvram --xformers
@REM --medvram --opt-split-attention --xformers
@REM --lowvram
@REM --lowvram --xformers
@REM --lowvram --opt-split-attention

call webui.bat
