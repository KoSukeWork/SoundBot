param(
    [string]$UserDataDir = (Join-Path $env:APPDATA 'soundbot'),
    [string]$ModelsPath = 'E:\SoundBot\resources\models',
    [int]$Port = 8011
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot '.venv-gpu313\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonExe)) { throw 'Run scripts/setup-gpu.ps1 first.' }
if (!(Test-Path -LiteralPath (Join-Path $ModelsPath 'clap\config.json'))) {
    throw 'CLAP model is missing. Pass -ModelsPath with the directory containing clap.'
}
# Read process names only. Never stop an installed/importing instance here.
$liveDataDir = [IO.Path]::GetFullPath((Join-Path $env:APPDATA 'soundbot')).TrimEnd('\')
$targetDataDir = [IO.Path]::GetFullPath($UserDataDir).TrimEnd('\')
if ($targetDataDir -ieq $liveDataDir) {
    $installedProcesses = Get-Process -Name 'SoundBot', 'soundbot-backend' -ErrorAction SilentlyContinue
    if ($installedProcesses) {
        throw 'The installed SoundBot is still running. Close it normally yourself before switching.'
    }
}
$env:SOUNDBOT_PYTHON = $pythonExe
$env:SOUNDBOT_USER_DATA_DIR = $targetDataDir
$env:SOUNDBOT_MODELS_PATH = [IO.Path]::GetFullPath($ModelsPath)
$env:SOUNDBOT_PORT = [string]$Port
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $repoRoot
try {
    & $pythonExe -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), 'PyTorch:', torch.__version__)"
    if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed' }
    npm run dev
} finally {
    Pop-Location
}
