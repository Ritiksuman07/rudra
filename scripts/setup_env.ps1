# RUDRA alpha v0.1 Setup Script (Windows PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RUDRA alpha v0.1 - Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create Python virtual environment
Write-Host "[1/5] Creating virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "  Done" -ForegroundColor Green
} else {
    Write-Host "  Already exists" -ForegroundColor Green
}

# Step 2: Activate and upgrade pip
Write-Host "[2/5] Upgrading pip..." -ForegroundColor Yellow
& .\venv\Scripts\pip install --upgrade pip
Write-Host "  Done" -ForegroundColor Green

# Step 3: Install PyTorch (with CUDA 11.8 support)
Write-Host "[3/5] Installing PyTorch + CUDA dependencies..." -ForegroundColor Yellow
& .\venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Write-Host "  Done" -ForegroundColor Green

# Step 4: Install ML libraries
Write-Host "[4/5] Installing ML libraries..." -ForegroundColor Yellow
& .\venv\Scripts\pip install `
    transformers>=4.45.0 `
    datasets>=3.0.0 `
    accelerate>=0.34.0 `
    peft>=0.13.0 `
    trl>=0.11.0 `
    bitsandbytes>=0.44.0 `
    sentencepiece `
    protobuf `
    huggingface-hub `
    wandb `
    tensorboard `
    pyyaml `
    jinja2 `
    requests `
    scipy `
    numpy `
    pandas
Write-Host "  Done" -ForegroundColor Green

# Step 5: Optional optimizations
Write-Host "[5/5] Installing optional optimizations..." -ForegroundColor Yellow
& .\venv\Scripts\pip install flash-attn --no-build-isolation
Write-Host "  Done" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "  Activate with: .\venv\Scripts\Activate" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Test installation
Write-Host ""
Write-Host "Testing PyTorch installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')"

Write-Host ""
Write-Host "Testing transformers installation..." -ForegroundColor Yellow
python -c "import transformers; print(f'transformers {transformers.__version__}')"