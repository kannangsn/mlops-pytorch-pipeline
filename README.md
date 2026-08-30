# mlops-pytorch-pipeline

End-to-end deployment pipeline for a PyTorch CIFAR-10 image classifier:
local development → containerized training with Docker → orchestrated
training and serving on Kubernetes.

**Course:** Machine Learning Operations Lab (DA5402W)
**Roll No:** DA25M574
**Name:** Kannan G S Nambiar 

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │           Kubernetes (ml-training ns)       │
                        │                                             │
 ┌──────────────┐       │  ┌────────────┐      ┌───────────────────┐  │
 │  GitHub repo │  CI   │  │ ConfigMap  │─────▶│  Training Job     │  │
 │  main/develop│─────▶ │  │ (training  │      │  (mlops-train:v1) │  │
 │  + PRs       │ build │  │  config)   │      └────────┬──────────┘  │
 └──────────────┘       │  └────────────┘               │ writes      │
                        │                               ▼             │
                        │  ┌────────────┐      ┌───────────────────┐  │
                        │  │  data-pvc  │      │  checkpoints-pvc  │  │
                        │  └────────────┘      └────────┬──────────┘  │
                        │                               │ read-only   │
                        │                               ▼             │
 ┌──────────────┐       │  ┌────────────┐      ┌───────────────────┐  │
 │    Client    │──────▶│  │  Service   │─────▶│ Serving Deployment│  │
 │ curl /predict│       │  │ (ClusterIP │      │ 2× mlops-serve:v1 │  │
 └──────────────┘       │  │  80→8080)  │      │ + liveness/ready  │  │
                        │  └────────────┘      │   probes + HPA    │  │
                        │                      └───────────────────┘  │
                        └─────────────────────────────────────────────┘
```

The training Job reads its hyperparameters from a ConfigMap mounted at
`/app/configs`, downloads CIFAR-10 into the `data-pvc` volume and writes the
best checkpoint to `checkpoints-pvc`. The serving Deployment mounts the same
checkpoint volume read-only and exposes `/predict` and `/health` behind a
ClusterIP Service, with an HPA scaling on CPU.

## Repository layout

```
src/            model.py, dataset.py, train.py, serve.py
configs/        training_config.yaml (full run), smoke_config.yaml (CI/tests)
docker/         Dockerfile.train (multi-stage), Dockerfile.serve (non-root)
k8s/            namespace, configmap, pvc, training-job, serving-deployment,
                serving-service, hpa
requirements/   train.txt, serve.txt, dev.txt (all versions pinned)
tests/          pytest suite for the model and the Flask API
run_tests.sh    one-shot setup + test + reproduce script (see below)
```

## Quick start: reproduce everything with one script

```bash
./run_tests.sh            # unit tests + smoke training + live serving check
./run_tests.sh --docker   # additionally builds both images and tests the
                          # serving container end-to-end
```

The script:

1. Checks for `python3` (3.10+), `python3-venv` and `curl`, and installs
   them via `apt` if missing (asks for sudo only in that case).
2. Creates a virtualenv at `.venv` and installs the pinned dependencies
   from `requirements/dev.txt` (CPU-only PyTorch wheels, ~600 MB download
   on the first run).
3. Runs the pytest suite in `tests/`.
4. Runs a 2-epoch smoke training on random ("fake") data — this exercises
   the real training loop, JSON-line logging, early-stopping bookkeeping and
   checkpointing without the 170 MB CIFAR-10 download.
5. Starts `src/serve.py` against the fresh checkpoint and hits
   `GET /health` and `POST /predict` with a generated test image.
6. With `--docker`: builds `mlops-train:v1` and `mlops-serve:v1`, runs the
   serving container with the checkpoint volume-mounted, and repeats the
   health/predict checks against the container on port 8080.

## Manual setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt     # or train.txt / serve.txt individually
pytest tests/ -v
```

### Local training (full CIFAR-10 run)

Edit `configs/training_config.yaml` so `data_dir` and `checkpoint_dir` point
somewhere writable (e.g. `./data`, `./checkpoints`), then:

```bash
python src/train.py --config configs/training_config.yaml
```

The config path can also come from the `CONFIG_PATH` env var or the default
`/app/configs/training_config.yaml` (used inside containers). Metrics are
printed as JSON lines; the best checkpoint (lowest val loss) is saved and
training stops early after `early_stopping_patience` epochs without
improvement.

### Local serving

```bash
MODEL_PATH=checkpoints/classifier_v1.pt python src/serve.py
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Docker

```bash
# Training image (multi-stage: deps built in a venv, copied into a slim runtime)
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

# Serving image (slim, inference-only deps, non-root user, HEALTHCHECK, port 8080)
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

To override the training config without rebuilding, mount a config directory
over the baked-in one: `-v $(pwd)/configs:/app/configs`.

## Kubernetes

Tested with minikube. Build the images into the cluster's Docker daemon so
`imagePullPolicy: IfNotPresent` finds them:

```bash
minikube start --cpus 2 --memory 4096 --driver=docker
eval $(minikube docker-env)
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

Deploy in order:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml

# watch training logs (JSON lines)
kubectl logs -f job/model-training -n ml-training

# once the job completes, bring up serving
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

# test the endpoint
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

Notes:

- The serving pods start even before a checkpoint exists; the readiness
  probe (`/health` returns 503 until the model loads) simply keeps them out
  of rotation, and the server re-tries loading on every health check.
- GPU bonus: `k8s/training-job-gpu.yaml` is a separate Job manifest with an
  `nvidia.com/gpu: 1` limit, a node selector and a toleration for GPU nodes.
  Apply it instead of `training-job.yaml` on a GPU-enabled cluster; on a
  CPU-only cluster it would stay Pending by design.
- The HPA needs metrics-server (`minikube addons enable metrics-server`).

## Git workflow

- `main` is the release branch, `develop` the integration branch.
- All work happens on feature branches (`feature/pytorch-model`,
  `feature/serving-api`, `feature/docker-images`,
  `feature/k8s-deployment`) merged into `develop` via PRs, then `develop → main`.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `docs:`,
  `test:`, `ci:`).
- No datasets, checkpoints or secrets are committed (see `.gitignore`);
  configuration is injected via ConfigMaps, and anything sensitive would go
  through Kubernetes Secrets / environment variables rather than the repo.

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main` and `develop`:
it installs the pinned dependencies, runs the pytest suite, does the fake-data
training smoke run, and then builds both Docker images.
