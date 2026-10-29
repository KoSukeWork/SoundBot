$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    uv venv .venv-gpu313 --python 3.13 --allow-existing
    if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
    uv --native-tls pip install --python .venv-gpu313/Scripts/python.exe -r backend/requirements.txt --overrides backend/requirements-gpu-overrides.txt --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed' }
    npm ci
    if ($LASTEXITCODE -ne 0) { throw 'Electron installation failed' }
    & .venv-gpu313/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
    if ($LASTEXITCODE -ne 0) { throw 'GPU check failed; check NVIDIA driver' }
} finally {
    Pop-Location
}
