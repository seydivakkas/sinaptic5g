# Windows PowerShell Smoke Test Utility for Sinaptic5G FTR Core
# OPEL LISANS - TUM HAKLAR SAKLIDIR
# Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)

param(
    [switch]$Build,
    [switch]$Gpu,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  SINAPTIK5G Windows PowerShell Smoke Test" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  -> Test dizinleri hazirlaniyor..."

New-Item -ItemType Directory -Force -Path "tests\smoke_input" | Out-Null
New-Item -ItemType Directory -Force -Path "tests\smoke_output" | Out-Null

$SyntheticVideo = "tests\smoke_input\video.mp4"
if (-not (Test-Path $SyntheticVideo)) {
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        ffmpeg -y -f lavfi -i testsrc=duration=4:size=640x480:rate=30 -c:v libx264 -pix_fmt yuv420p "$SyntheticVideo" 2>$null
        Write-Host "  [OK] Sentetik test videosu olusturuldu (4s, 640x480)" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] ffmpeg bulunamadi. Lutfen local olarak 'tests\smoke_input\video.mp4' yoluna bir mp4 video koyun." -ForegroundColor Red
        Exit 1
    }
} else {
    Write-Host "  [OK] Sentetik test videosu hazir." -ForegroundColor Green
}

# Phase 1: Docker Build
$ImageName = "sinaptik5g"
$ImageTag = "latest"

$ImageExists = docker images -q "$ImageName`:$ImageTag"

if ($Build -or -not $ImageExists) {
    Write-Host "  -> Docker image build ediliyor..." -ForegroundColor Yellow
    docker build -t "$ImageName`:$ImageTag" -f docker/Dockerfile.ftr .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] docker build basarisiz oldu! Kod: $LASTEXITCODE" -ForegroundColor Red
        Exit 1
    }
    Write-Host "  [OK] docker build basarili: $ImageName`:$ImageTag" -ForegroundColor Green
} else {
    Write-Host "  [OK] Docker imaji hazir: $ImageName`:$ImageTag (build atlandı)" -ForegroundColor Green
}

# Phase 2: Docker Run
if (Test-Path "tests\smoke_output\results.json") {
    Remove-Item "tests\smoke_output\results.json" -Force
}

$CurrentDir = Get-Location
$InputMount = "$CurrentDir\tests\smoke_input"
$OutputMount = "$CurrentDir\tests\smoke_output"

Write-Host "  -> Container calistiriliyor..." -ForegroundColor Yellow

$DockerArgs = @("run", "--rm", "-v", "$InputMount`:/app/data/input:ro", "-v", "$OutputMount`:/app/data/output:rw", "--network", "none")
if ($Gpu) {
    $DockerArgs += "--gpus"
    $DockerArgs += "all"
}
$DockerArgs += "$ImageName`:$ImageTag"

& docker $DockerArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Container hata ile sonlandi! Kod: $LASTEXITCODE" -ForegroundColor Red
    Exit 1
}

Write-Host "  [OK] Container basariyla tamamlandi (exit 0)" -ForegroundColor Green

# Phase 3: Output Controls
$ResultsJson = "tests\smoke_output\results.json"
if (-not (Test-Path $ResultsJson)) {
    Write-Host "  [ERROR] results.json uretilmedi!" -ForegroundColor Red
    Exit 1
}
Write-Host "  [OK] results.json uretildi" -ForegroundColor Green

# Check if JSON is valid
try {
    $RawJson = Get-Content $ResultsJson -Raw -Encoding UTF8
    $Parsed = ConvertFrom-Json $RawJson
    Write-Host "  [OK] results.json gecerli JSON formatinda" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] results.json gecersiz JSON formati!" -ForegroundColor Red
    Exit 1
}

# Check required keys
if (-not ($Parsed.PSObject.Properties.Name -contains "video_id" -and $Parsed.PSObject.Properties.Name -contains "arac_bilgisi" -and $Parsed.PSObject.Properties.Name -contains "tespitler")) {
    Write-Host "  [ERROR] Zorunlu alanlar (video_id, arac_bilgisi, tespitler) eksik!" -ForegroundColor Red
    Exit 1
}
Write-Host "  [OK] Zorunlu alanlar: OK" -ForegroundColor Green

# Category + confidence range check
$Valid = $true
if ($Parsed.arac_bilgisi.confidence_score -lt 0.0 -or $Parsed.arac_bilgisi.confidence_score -gt 1.0) {
    $Valid = $false
}
foreach ($det in $Parsed.tespitler) {
    if ($det.confidence_score -lt 0.0 -or $det.confidence_score -gt 1.0) {
        $Valid = $false
    }
    if ($det.kategori -notin @("sofor_eylemi", "nesneler", "yolcular")) {
        Write-Host "  [ERROR] Gecersiz kategori '$($det.kategori)'!" -ForegroundColor Red
        $Valid = $false
    }
}

if (-not $Valid) {
    Write-Host "  [ERROR] Kategori veya confidence score sinir deger disi!" -ForegroundColor Red
    Exit 1
}
Write-Host "  [OK] Kategori + confidence kontrolleri: OK" -ForegroundColor Green

# Phase 4: Model Integrity Lock Checks
try {
    $Lock = ConvertFrom-Json (Get-Content "model_lock.json" -Raw -Encoding UTF8)
    
    # Helper to calculate SHA256
    function Get-FileHashSha256($Path) {
        $Hash = Get-FileHash -Path $Path -Algorithm SHA256
        return $Hash.Hash.ToLower()
    }
    
    $OptHash = Get-FileHashSha256 "models\detector_optimized.onnx"
    if ($OptHash -eq $Lock.detector_optimized_onnx_sha256) {
        Write-Host "  [OK] SHA-256 OK: detector_optimized" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] detector_optimized.onnx SHA-256 butunluk dogrulamasi uyusmuyor!" -ForegroundColor Red
        Exit 1
    }
} catch {
    Write-Host "  [ERROR] Model hash dogrulamasi sirasinda hata olustu!" -ForegroundColor Red
    Exit 1
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  [SUCCESS] TUM TESTLER GECTI: 8/8" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

# Clean after test
if ($Clean) {
    Remove-Item "tests\smoke_input" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item "tests\smoke_output" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  [OK] Test klasorleri temizlendi." -ForegroundColor Green
}
