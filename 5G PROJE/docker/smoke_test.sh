#!/bin/bash
# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
#
# Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
# amaçlı olarak paylaşılmıştır.

set -e

# Help command helper
show_help() {
    echo "SİNAPTİK5G FTR Core Smoke Test Utility"
    echo "Usage: bash tests/smoke_test.sh [options]"
    echo ""
    echo "Options:"
    echo "  --build      Re-build the Docker container before running the test."
    echo "  --gpu        Run the Docker container with GPU access enabled (--gpus all)."
    echo "  --clean      Clean generated input/output test files after running tests."
    echo "  --help       Show this help message."
    exit 0
}

# Parse command line arguments
BUILD_IMAGE=false
USE_GPU=false
CLEAN_AFTER=false

for arg in "$@"; do
    case $arg in
        --build)
        BUILD_IMAGE=true
        shift
        ;;
        --gpu)
        USE_GPU=true
        shift
        ;;
        --clean)
        CLEAN_AFTER=true
        shift
        ;;
        --help)
        show_help
        ;;
    esac
done

echo "════════════════════════════════════════════════════════"
echo "  SİNAPTİK5G Docker Smoke Test"
echo "════════════════════════════════════════════════════════"
echo "  → Test dizinleri hazırlanıyor..."

mkdir -p tests/smoke_input tests/smoke_output

# Generate 4s synthetic test video if not exists
SYNTHETIC_VIDEO="tests/smoke_input/video.mp4"
if [ ! -f "$SYNTHETIC_VIDEO" ]; then
    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -y -f lavfi -i testsrc=duration=4:size=640x480:rate=30 -c:v libx264 -pix_fmt yuv420p "$SYNTHETIC_VIDEO" >/dev/null 2>&1
        echo "  ✓ Sentetik test videosu oluşturuldu (4s, 640×480)"
    else
        echo "  [-] Uyarı: ffmpeg bulunamadı. Sentetik test videosu oluşturulamıyor."
        echo "  [-] Lütfen 'tests/smoke_input/video.mp4' dosyası olarak örnek bir mp4 yerleştirin."
        exit 1
    fi
else
    echo "  ✓ Sentetik test videosu zaten hazır."
fi

# Docker Build
IMAGE_NAME="sinaptik5g"
IMAGE_TAG="latest"

if [ "$BUILD_IMAGE" = true ]; then
    echo "  → Docker image build ediliyor..."
    docker build -t "$IMAGE_NAME:$IMAGE_TAG" -f docker/Dockerfile.ftr . >/dev/null
    echo "  ✓ docker build başarılı: $IMAGE_NAME:$IMAGE_TAG"
else
    if ! docker image inspect "$IMAGE_NAME:$IMAGE_TAG" >/dev/null 2>&1; then
        echo "  → Görüntü mevcut değil, build ediliyor..."
        docker build -t "$IMAGE_NAME:$IMAGE_TAG" -f docker/Dockerfile.ftr . >/dev/null
        echo "  ✓ docker build başarılı: $IMAGE_NAME:$IMAGE_TAG"
    else
        echo "  ✓ Docker imajı hazır: $IMAGE_NAME:$IMAGE_TAG (build atlandı)"
    fi
fi

# Run Container
rm -f tests/smoke_output/results.json

INPUT_MOUNT="$(pwd)/tests/smoke_input"
OUTPUT_MOUNT="$(pwd)/tests/smoke_output"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    INPUT_MOUNT=$(cygpath -w "$INPUT_MOUNT")
    OUTPUT_MOUNT=$(cygpath -w "$OUTPUT_MOUNT")
fi

echo "  → Container çalıştırılıyor..."
DOCKER_OPTS="--rm -v $INPUT_MOUNT:/app/data/input:ro -v $OUTPUT_MOUNT:/app/data/output:rw --network none"

if [ "$USE_GPU" = true ]; then
    docker run $DOCKER_OPTS --gpus all "$IMAGE_NAME:$IMAGE_TAG" >/dev/null
else
    docker run $DOCKER_OPTS "$IMAGE_NAME:$IMAGE_TAG" >/dev/null
fi

echo "  ✓ Container başarıyla tamamlandı (exit 0)"

# Controls
RESULTS_JSON="tests/smoke_output/results.json"
if [ ! -f "$RESULTS_JSON" ]; then
    echo "  ✗ HATA: results.json üretilmedi!"
    exit 1
fi
echo "  ✓ results.json üretildi"

if ! python3 -c "import json; json.load(open('$RESULTS_JSON'))" >/dev/null 2>&1; then
    echo "  ✗ HATA: results.json geçersiz JSON formatı!"
    exit 1
fi
echo "  ✓ results.json geçerli JSON formatında"

REQUIRED_CHECK=$(python3 -c "
import json
data = json.load(open('$RESULTS_JSON'))
keys = ['video_id', 'arac_bilgisi', 'tespitler']
if all(k in data for k in keys):
    print('OK')
else:
    print('FAILED')
")

if [ "$REQUIRED_CHECK" != "OK" ]; then
    echo "  ✗ HATA: Zorunlu alanlar (video_id, arac_bilgisi, tespitler) eksik!"
    exit 1
fi
echo "  ✓ Zorunlu alanlar: OK"

# JSON schema validation using our schemas/results.schema.json
SCHEMA_CHECK=$(python3 -c "
import json
from jsonschema import Draft202012Validator
schema = json.load(open('schemas/results.schema.json'))
data = json.load(open('$RESULTS_JSON'))
try:
    Draft202012Validator(schema).validate(data)
    print('OK')
except Exception as e:
    print('FAILED:', e)
")

if [ "$SCHEMA_CHECK" != "OK" ]; then
    echo "  ✗ HATA: JSON şema doğrulaması başarısız: $SCHEMA_CHECK"
    exit 1
fi
echo "  ✓ JSON şema doğrulaması geçti"

# Clean after test
if [ "$CLEAN_AFTER" = true ]; then
    rm -rf tests/smoke_input tests/smoke_output
    echo "  ✓ Test klasörleri temizlendi."
fi

echo "════════════════════════════════════════════════════════"
echo "  ✅ TÜM SMOKE TEST KONTROLLERİ GEÇTİ"
echo "════════════════════════════════════════════════════════"
