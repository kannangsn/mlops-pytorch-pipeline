#!/usr/bin/env bash
#
# run_tests.sh - one-shot script to set everything up and reproduce results.
#
#   ./run_tests.sh            unit tests + smoke training + live serving check
#   ./run_tests.sh --docker   also build both Docker images and test the
#                             serving container end-to-end (needs Docker)
#
# What it does:
#   1. checks/installs system packages (python3, venv, curl) via apt if missing
#   2. creates a virtualenv at .venv and installs pinned dependencies
#   3. runs the pytest suite (tests/)
#   4. runs a short training smoke run (fake data, 2 epochs) -> checkpoint
#   5. starts the Flask server against that checkpoint and hits
#      /health and /predict with a generated test image
#
set -euo pipefail
cd "$(dirname "$0")"

RUN_DOCKER=0
[[ "${1:-}" == "--docker" ]] && RUN_DOCKER=1

step() { echo; echo "==> $*"; }

# ---------------------------------------------------------------- 1. system deps
step "Checking system prerequisites"
missing=()
command -v python3 >/dev/null || missing+=(python3)
command -v curl    >/dev/null || missing+=(curl)
python3 -m venv --help >/dev/null 2>&1 || missing+=(python3-venv)

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Installing missing packages: ${missing[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y -qq "${missing[@]}" python3-pip
else
    echo "python3, venv and curl are available"
fi

pyver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $pyver"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || { echo "ERROR: Python 3.10+ required"; exit 1; }

# ---------------------------------------------------------------- 2. virtualenv
step "Setting up virtualenv (.venv) and installing pinned dependencies"
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements/dev.txt
python - <<'EOF'
import torch, torchvision
from importlib.metadata import version
print(f"Installed: torch {torch.__version__}, "
      f"torchvision {torchvision.__version__}, flask {version('flask')}")
EOF

# ---------------------------------------------------------------- 3. unit tests
step "Running unit tests"
pytest tests/ -v

# ---------------------------------------------------------------- 4. smoke training
step "Smoke training run (fake data, 2 epochs, checkpoint to ./checkpoints)"
rm -f checkpoints/smoke_model.pt
python src/train.py --config configs/smoke_config.yaml
[[ -f checkpoints/smoke_model.pt ]] \
    || { echo "ERROR: training did not produce a checkpoint"; exit 1; }
echo "Checkpoint written: $(ls -lh checkpoints/smoke_model.pt | awk '{print $5, $9}')"

# ---------------------------------------------------------------- 5. serving check
step "Starting the model server and testing /health and /predict"
python - <<'EOF'
from PIL import Image
Image.new("RGB", (32, 32), color=(90, 140, 60)).save("test_image.png")
EOF

MODEL_PATH=checkpoints/smoke_model.pt PORT=8081 python src/serve.py >server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 20); do
    sleep 1
    if curl -s -o /dev/null http://127.0.0.1:8081/health; then break; fi
    [[ $i -eq 20 ]] && { echo "ERROR: server did not start"; cat server.log; exit 1; }
done

echo "GET /health:"
curl -s -w '  -> HTTP %{http_code}\n' http://127.0.0.1:8081/health
echo "POST /predict:"
curl -s -w '\n  -> HTTP %{http_code}\n' -X POST http://127.0.0.1:8081/predict \
    -F "image=@test_image.png"

kill $SERVER_PID 2>/dev/null || true
trap - EXIT

# ---------------------------------------------------------------- 6. docker (optional)
if [[ $RUN_DOCKER -eq 1 ]]; then
    step "Building Docker images"
    docker build -f docker/Dockerfile.train -t mlops-train:v1 .
    docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

    step "Testing the serving container against the smoke checkpoint"
    cp checkpoints/smoke_model.pt checkpoints/classifier_v1.pt
    docker rm -f mlops-serve-test >/dev/null 2>&1 || true
    docker run -d --name mlops-serve-test -p 8080:8080 \
        -v "$(pwd)/checkpoints:/app/checkpoints:ro" mlops-serve:v1

    for i in $(seq 1 30); do
        sleep 1
        if curl -s -o /dev/null http://127.0.0.1:8080/health; then break; fi
        [[ $i -eq 30 ]] && { echo "ERROR: container did not become healthy"; \
                             docker logs mlops-serve-test; docker rm -f mlops-serve-test; exit 1; }
    done

    echo "GET /health (container):"
    curl -s -w '  -> HTTP %{http_code}\n' http://127.0.0.1:8080/health
    echo "POST /predict (container):"
    curl -s -w '\n  -> HTTP %{http_code}\n' -X POST http://127.0.0.1:8080/predict \
        -F "image=@test_image.png"

    docker rm -f mlops-serve-test >/dev/null
fi

step "All checks passed"
