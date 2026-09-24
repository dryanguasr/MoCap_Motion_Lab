param(
    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'
$mainPython = Join-Path $RepositoryRoot '.venv\Scripts\python.exe'
$uv = Join-Path $RepositoryRoot '.venv\Scripts\uv.exe'
$environment = Join-Path $RepositoryRoot '.venv_racketvision'
$source = Join-Path $RepositoryRoot '.cache\racketvision\source'
$commit = 'c44af2a08524d3cb54d818f19686f4cdea4d2793'

if (-not (Test-Path -LiteralPath $uv)) {
    & $mainPython -m pip install 'uv>=0.8,<0.9'
}
if (-not (Test-Path -LiteralPath $source)) {
    git clone --filter=blob:none https://github.com/OrcustD/RacketVision.git $source
}
git -C $source fetch origin $commit
git -C $source checkout --detach $commit
& $uv python install 3.10
& $uv venv --python 3.10 $environment
$python = Join-Path $environment 'Scripts\python.exe'

& $uv pip install --python $python torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
& $uv pip install --python $python numpy==1.26.4 opencv-python==4.10.0.84 opencv-python-headless==4.10.0.84 pandas tqdm scikit-learn parse huggingface-hub openmim mmengine==0.10.7 albumentations json-tricks munkres xtcocotools
& $uv pip install --python $python mmcv==2.1.0 --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
& $uv pip install --python $python mmdet==3.2.0
& $uv pip install --python $python setuptools==80.9.0
& $python -m pip install chumpy==0.70 --no-build-isolation
& $uv pip install --python $python mmpose==1.3.2 --no-deps

$upstreamSource = Join-Path $source 'source'
Push-Location $upstreamSource
try {
    & $python download_checkpoints.py --module BallTrack
    & $python download_checkpoints.py --module RacketPose
} finally {
    Pop-Location
}
& $python -c "import mmengine,mmcv,mmdet,mmpose,torch; print(mmengine.__version__,mmcv.__version__,mmdet.__version__,mmpose.__version__,'cuda',torch.cuda.is_available())"
