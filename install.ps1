<#
.SYNOPSIS
    Устанавливает llama-cpp-python под конкретное железо (CPU или CUDA),
    с автоматическим фолбэком на CPU, если CUDA-сборка недоступна или
    не устанавливается.

.DESCRIPTION
    Вынесено в отдельный скрипт от requirements.txt, потому что для этого
    пакета нет одной "правильной" строчки версии: нужный wheel зависит от
    версии Python, наличия и версии CUDA. Подробности - см. README.md,
    раздел "Диагностика проблем установки" (там же - какие именно ошибки
    этот скрипт призван предотвратить).

.PARAMETER Cpu
    Принудительно ставить CPU-сборку, даже если обнаружена NVIDIA GPU.

.PARAMETER CudaVersion
    Переопределить автоопределение версии CUDA, например "cu124" или "cu130".
    Список поддерживаемых индексов см. в выводе `nvidia-smi` (поле
    "CUDA Version") и на https://github.com/abetlen/llama-cpp-python#supported-backends

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Cpu
    .\install.ps1 -CudaVersion cu124
#>
param(
    [switch]$Cpu,
    [string]$CudaVersion
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# --- 1. Проверка версии Python (Проблемы №1, №9) ---------------------------
Write-Step "Проверка версии Python"
$verOutput = python --version 2>&1
Write-Host $verOutput
if ($verOutput -notmatch "Python 3\.(1[0-2])\.") {
    Write-Host "ОШИБКА: нужен Python 3.10, 3.11 или 3.12 (numpy/torch/llama-cpp-python " -ForegroundColor Red
    Write-Host "публикуют готовые wheel только под эти версии). Активируйте правильный venv:" -ForegroundColor Red
    Write-Host "    py -3.11 -m venv venv" -ForegroundColor Yellow
    Write-Host "    venv\Scripts\activate" -ForegroundColor Yellow
    exit 1
}

# --- 2. Основные зависимости -------------------------------------------
Write-Step "Установка requirements.txt (без llama-cpp-python)"
python -m pip install --upgrade pip
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ОШИБКА при установке requirements.txt - см. вывод выше." -ForegroundColor Red
    exit 1
}

# --- 3. Определение GPU/CUDA --------------------------------------------
function Get-CudaIndexTag {
    if ($Cpu) { return $null }
    if ($CudaVersion) { return $CudaVersion }

    try {
        $smi = nvidia-smi 2>&1
    } catch {
        return $null  # nvidia-smi не найден - GPU нет или драйвер не установлен
    }
    if ($smi -notmatch "CUDA Version:\s*([\d\.]+)") {
        return $null
    }
    $detected = $Matches[1]
    Write-Host "Обнаружена NVIDIA GPU, версия CUDA драйвера: $detected"

    # Карта поддерживаемых предсобранных индексов abetlen/llama-cpp-python.
    # Список может обновляться - актуальный см. в README проекта llama-cpp-python.
    $known = @("13.2", "13.0", "12.5", "12.4", "12.3", "12.2", "12.1", "11.8")
    $tags  = @{ "13.2"="cu132"; "13.0"="cu130"; "12.5"="cu125"; "12.4"="cu124";
                "12.3"="cu123"; "12.2"="cu122"; "12.1"="cu121"; "11.8"="cu118" }

    $detectedVer = [version]$detected
    foreach ($k in $known) {
        if ($detectedVer -ge [version]$k) { return $tags[$k] }
    }
    return $null
}

$cudaTag = Get-CudaIndexTag

# --- 4. Установка llama-cpp-python с фолбэком на CPU ------------------------
function Install-LlamaCppPython($indexTag) {
    if ($indexTag) {
        Write-Step "Установка llama-cpp-python (CUDA: $indexTag)"
        pip install llama-cpp-python==0.3.35 --prefer-binary `
            --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$indexTag"
    } else {
        Write-Step "Установка llama-cpp-python (CPU)"
        pip install llama-cpp-python==0.3.35 --prefer-binary `
            --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cpu"
    }
    return $LASTEXITCODE -eq 0
}

$installed = $false
if ($cudaTag) {
    $installed = Install-LlamaCppPython $cudaTag
    if (-not $installed) {
        Write-Host "CUDA-сборка ($cudaTag) не установилась, пробую CPU-сборку..." -ForegroundColor Yellow
    }
}
if (-not $installed) {
    $installed = Install-LlamaCppPython $null
}
if (-not $installed) {
    Write-Host "`nНе удалось поставить готовый wheel ни для GPU, ни для CPU." -ForegroundColor Red
    Write-Host "Вариант 2 из README ('Диагностика проблем установки'): поставить" -ForegroundColor Red
    Write-Host "Visual Studio Build Tools (C++) и собрать пакет из исходников." -ForegroundColor Red
    exit 1
}

# --- 5. Проверка, что GPU реально используется (Проблема №11) --------------
Write-Step "Проверка поддержки GPU в установленной сборке"
$check = python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())" 2>&1
Write-Host $check
if ($cudaTag -and ($check -notmatch "True")) {
    Write-Host "`nВНИМАНИЕ: GPU была обнаружена, но установленная сборка llama-cpp-python " -ForegroundColor Yellow
    Write-Host "не поддерживает офлоад (или не смогла определить это). Модель будет " -ForegroundColor Yellow
    Write-Host "работать на CPU. Известная проблема Windows-сборок - см. README, " -ForegroundColor Yellow
    Write-Host "раздел 'Использование GPU', пункт про альтернативный wheel." -ForegroundColor Yellow
}

Write-Step "Готово"
Write-Host "Дальше: python prepare.py, затем python run.py --out ./out"
