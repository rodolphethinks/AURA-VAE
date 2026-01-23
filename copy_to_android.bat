@echo off
REM Copy TFLite models to Android assets

set SOURCE_DIR=%~dp0models
set TARGET_DIR=%~dp0android\AuraVAE\app\src\main\assets

echo Copying TFLite models to Android assets...

if not exist "%SOURCE_DIR%\vae_model.tflite" (
    echo Error: vae_model.tflite not found in models directory
    echo Run the training pipeline first: run.bat
    pause
    exit /b 1
)

REM Create target directory if it doesn't exist
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

REM Copy files
echo Copying vae_model.tflite...
copy "%SOURCE_DIR%\vae_model.tflite" "%TARGET_DIR%\" /Y

if exist "%SOURCE_DIR%\anomaly_detector.tflite" (
    echo Copying anomaly_detector.tflite...
    copy "%SOURCE_DIR%\anomaly_detector.tflite" "%TARGET_DIR%\" /Y
)

if exist "%SOURCE_DIR%\android_config.json" (
    echo Copying android_config.json...
    copy "%SOURCE_DIR%\android_config.json" "%TARGET_DIR%\" /Y
)

if exist "%SOURCE_DIR%\threshold_config.json" (
    echo Copying threshold_config.json...
    copy "%SOURCE_DIR%\threshold_config.json" "%TARGET_DIR%\" /Y
)

if exist "%SOURCE_DIR%\normalization_params.json" (
    echo Copying normalization_params.json...
    copy "%SOURCE_DIR%\normalization_params.json" "%TARGET_DIR%\" /Y
)

echo Done!

copy "%SOURCE_DIR%\anomaly_detector.tflite" "%TARGET_DIR%\" /Y
copy "%SOURCE_DIR%\android_config.json" "%TARGET_DIR%\" /Y

echo.
echo Files copied successfully to:
echo %TARGET_DIR%
echo.
echo Now open the Android project in Android Studio and build.
pause
