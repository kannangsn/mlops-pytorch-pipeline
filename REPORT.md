# Assignment Report — Deploying PyTorch ML Workloads with Docker & Kubernetes

**Course:** DA5402W - Machine Learning Operations Lab
**Roll No:** DA25M574

- GitHub repository: `https://github.com/kannangsn/mlops-pytorch-pipeline`
- Final validation PR: `https://github.com/kannangsn/mlops-pytorch-pipeline/pull/5`

---

## 1. What was built

| Part | Deliverable | Where |
|------|-------------|-------|
| A | Repo structure, `.gitignore`, CI workflow, Git branching plan | repo root, `.github/workflows/ci.yml` |
| B | ResNet-18 (CIFAR-adapted) + SimpleCNN, CIFAR-10 loaders, config-driven training loop with JSON-line logging and early stopping, Flask serving app | `src/` |
| C | Multi-stage training image, slim non-root serving image with `HEALTHCHECK` | `docker/` |
| D | Namespace, ConfigMap, PVCs, training Job; GPU bonus as a separate Job manifest | `k8s/` |
| E | Serving Deployment (2 replicas, probes, rolling update), ClusterIP Service, HPA | `k8s/` |
| F | End-to-end validation steps and logs | this report + `run_tests.sh` |

Key Design decisions:

- **Config injection.** `train.py` resolves its config from `--config`, then the
  `CONFIG_PATH` env var, then `/app/configs/training_config.yaml` — so the same
  image works locally (mounted volume), in Docker and under Kubernetes where the
  ConfigMap is mounted at `/app/configs`.
- **Checkpoint self-description.** The checkpoint stores the architecture name
  and class count, so `serve.py` reconstructs the right model without a second
  config file that could drift out of sync.
- **Serving is checkpoint-tolerant.** If the checkpoint does not exist yet
  (training Job still running), the server starts anyway and `/health` returns
  503; the readiness probe keeps the pod out of rotation and the server retries
  loading on each health check. Once training writes the file to the shared PVC,
  the pods become ready without a restart.
- **A `fake` dataset option** (torchvision `FakeData`) lets the unit tests and
  CI exercise the full training loop without downloading 170 MB of CIFAR-10.
- **CPU-only PyTorch wheels** in the pinned requirements keep the Docker images
  and CI installs several GB smaller than the default CUDA build.

### Conformance with course clarifications (Aug 2026 mailing-list thread)

- The "2 PRs per week" wording was clarified as a suggestion of ~4 PRs total,
  acceptable within a short duration — this repo has 4 feature PRs plus a
  release PR.
- PVCs are defined in a separate `k8s/pvc.yaml`, as explicitly approved.
- The GPU bonus is implemented as a separate manifest
  (`k8s/training-job-gpu.yaml`) rather than commented-out configuration, per
  the clarification; the `nodeSelector` label (`accelerator: nvidia-gpu`) is
  self-chosen as permitted.
- `ci.yml` and `hpa.yaml` were clarified as not required; both are included
  anyway as working extras (CI runs the test suite and Docker builds on every
  PR; the HPA scales the serving deployment on CPU).
- `tests/test_model.py` is implemented with real tests, as required.

## 2. Reproducing the results

Everything below is reproduced by one script (see `README.md` for details):

```bash
./run_tests.sh            # deps + unit tests + smoke training + serving check
./run_tests.sh --docker   # additionally builds and tests the Docker images
```

### 2.1 Unit tests

```
$ pytest tests/ -v
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.2.2, pluggy-1.6.0
collected 11 items

tests/test_model.py::test_simple_cnn_output_shape PASSED                 [  9%]
tests/test_model.py::test_resnet18_output_shape PASSED                   [ 18%]
tests/test_model.py::test_unknown_architecture_raises PASSED             [ 27%]
tests/test_model.py::test_softmax_gives_valid_probabilities PASSED       [ 36%]
tests/test_model.py::test_checkpoint_roundtrip PASSED                    [ 45%]
tests/test_model.py::test_training_config_is_complete PASSED             [ 54%]
tests/test_serve.py::test_health_ok_when_model_loaded PASSED             [ 63%]
tests/test_serve.py::test_health_503_when_checkpoint_missing PASSED      [ 72%]
tests/test_serve.py::test_predict_returns_probabilities PASSED           [ 81%]
tests/test_serve.py::test_predict_rejects_missing_file PASSED            [ 90%]
tests/test_serve.py::test_predict_rejects_non_image PASSED               [100%]

============================== 11 passed in 2.23s ==============================
```

### 2.2 Training run (JSON-line metrics, early stopping, checkpointing)

Short CIFAR-10 run (ResNet-18, 3 epochs, 120 batches/epoch on CPU):

```
$ python src/train.py --config <config with dataset=cifar10, epochs=3, max_batches=120>
{"event": "config_loaded", "path": ".../training_config.yaml"}
{"event": "device_selected", "device": "cpu"}
{"epoch": 1, "train_loss": 1.8144, "train_accuracy": 0.3302, "val_loss": 1.6553, "val_accuracy": 0.3625}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt"}
{"epoch": 2, "train_loss": 1.5143, "train_accuracy": 0.4396, "val_loss": 1.604, "val_accuracy": 0.4188}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt"}
{"epoch": 3, "train_loss": 1.3508, "train_accuracy": 0.5069, "val_loss": 1.4691, "val_accuracy": 0.5122}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt"}
{"event": "training_complete", "best_val_loss": 1.4691}
```

The loss falls and accuracy rises epoch over epoch, and the best checkpoint is
re-saved whenever validation loss improves, which is the behaviour the early
stopping logic is built on.

### 2.3 Docker verification (Part C)

Both images build from the same repo root context; the serving container was
then run with the checkpoint directory mounted read-only:

```
$ docker build -f docker/Dockerfile.train -t mlops-train:v1 .
 => naming to docker.io/library/mlops-train:v1                    DONE 77.8s
$ docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
 => naming to docker.io/library/mlops-serve:v1                    DONE 65.1s

$ docker run -d --name mlops-serve-test -p 8080:8080 \
    -v $(pwd)/checkpoints:/app/checkpoints:ro mlops-serve:v1

$ curl -s http://127.0.0.1:8080/health
{"model_path":"/app/checkpoints/classifier_v1.pt","status":"ok"}   # HTTP 200

# test_image.png is a frog from the CIFAR-10 test set
$ curl -s -X POST http://127.0.0.1:8080/predict -F "image=@test_image.png"
{"confidence":0.9051,"predicted_class":"frog","probabilities":{"airplane":0.0001,
"automobile":0.0006,"bird":0.0659,"cat":0.0132,"deer":0.0084,"dog":0.0013,
"frog":0.9051,"horse":0.0004,"ship":0.0047,"truck":0.0003}}        # HTTP 200

$ docker inspect --format '{{.State.Health.Status}}' mlops-serve-test
healthy                                  # HEALTHCHECK instruction working
$ docker exec mlops-serve-test whoami
appuser                                  # non-root user
```

The training container was also verified with the config supplied from a
mounted volume (the same mechanism the ConfigMap uses on Kubernetes):

```
$ docker run --rm \
    -v $(pwd)/mounted-config:/app/configs:ro \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    mlops-train:v1
{"event": "config_loaded", "path": "/app/configs/training_config.yaml"}
{"event": "device_selected", "device": "cpu"}
{"epoch": 1, "train_loss": 2.9216, "train_accuracy": 0.0833, "val_loss": 2.3032, "val_accuracy": 0.1875}
{"event": "checkpoint_saved", "path": "/app/checkpoints/docker_smoke.pt"}
{"event": "training_complete", "best_val_loss": 2.3032}
```

### 2.4 Kubernetes validation (Parts D–F)

Commands used on minikube (also in `README.md`):

```bash
minikube start --cpus 4 --memory 8192
eval $(minikube docker-env)
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml
kubectl logs -f job/model-training -n ml-training

kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

Full terminal evidence (11 screenshots, `Screenshots/SS-1` through `SS-11`,
also attached to PR #5) is transcribed below.

**A bug was found and fixed during this validation.** The first deploy of
`serving-deployment.yaml` crash-looped: both pods restarted 5 times before
stabilizing (`kubectl describe pod` showed the liveness probe killing the
container with `connection refused` / `context deadline exceeded`). Root
cause: the container needs 30-55s on cold start to import
torch/torchvision and load the checkpoint off the PVC, but the liveness
probe had no `initialDelaySeconds`, so kubelet started killing it before it
was ready. Fixed by adding `initialDelaySeconds: 45` and `timeoutSeconds: 5`
to the liveness probe (commit `d774d71`). Screenshots 8-9 below reflect the
fixed, redeployed pods.

```
$ minikube start --cpus 2 --memory 4096 --driver=docker
...
Done! kubectl is now configured to use "minikube" cluster and "default" namespace by default
$ kubectl get nodes
NAME       STATUS   ROLES           AGE    VERSION
minikube   Ready    control-plane   3m8s   v1.35.1

$ kubectl apply -f k8s/namespace.yaml
namespace/ml-training created
$ kubectl apply -f k8s/configmap.yaml
configmap/training-config created
$ kubectl apply -f k8s/pvc.yaml
persistentvolumeclaim/data-pvc created
persistentvolumeclaim/checkpoints-pvc created
$ kubectl apply -f k8s/training-job.yaml
job.batch/model-training created

$ kubectl logs -f job/model-training -n ml-training
{"event": "config_loaded", "path": "/app/configs/training_config.yaml"}
{"event": "device_selected", "device": "cpu"}
Downloading https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz to /app/data/cifar-10-python.tar.gz
100.0%
Extracting /app/data/cifar-10-python.tar.gz to /app/data
Files already downloaded and verified
{"epoch": 1, "train_loss": 1.3356, "train_accuracy": 0.5144, "val_loss": 1.7182, "val_accuracy": 0.5011}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 2, "train_loss": 0.8562, "train_accuracy": 0.7007, "val_loss": 0.9004, "val_accuracy": 0.6983}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 3, "train_loss": 0.6821, "train_accuracy": 0.7625, "val_loss": 0.6348, "val_accuracy": 0.7814}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 4, "train_loss": 0.5837, "train_accuracy": 0.7993, "val_loss": 0.5894, "val_accuracy": 0.8016}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 5, "train_loss": 0.503, "train_accuracy": 0.8276, "val_loss": 0.5992, "val_accuracy": 0.8002}
{"epoch": 6, "train_loss": 0.4462, "train_accuracy": 0.8458, "val_loss": 0.5122, "val_accuracy": 0.8313}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 7, "train_loss": 0.4009, "train_accuracy": 0.8618, "val_loss": 0.4742, "val_accuracy": 0.845}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 8, "train_loss": 0.3699, "train_accuracy": 0.8732, "val_loss": 0.4123, "val_accuracy": 0.8594}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 9, "train_loss": 0.3316, "train_accuracy": 0.8856, "val_loss": 0.4161, "val_accuracy": 0.857}
{"epoch": 10, "train_loss": 0.3096, "train_accuracy": 0.8929, "val_loss": 0.3954, "val_accuracy": 0.8714}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"event": "training_complete", "best_val_loss": 0.3954}

$ kubectl get jobs -n ml-training
NAME             STATUS     COMPLETIONS   DURATION   AGE
model-training   Complete   1/1           9h         9h

$ kubectl apply -f k8s/serving-deployment.yaml
deployment.apps/model-serving created
$ kubectl apply -f k8s/serving-service.yaml
service/model-serving created
$ kubectl apply -f k8s/hpa.yaml
horizontalpodautoscaler.autoscaling/model-serving-hpa created

$ kubectl get pods -n ml-training
NAME                            READY   STATUS      RESTARTS   AGE
model-serving-5bb9c5df5-bmplt   1/1     Running     0          2m55s
model-serving-5bb9c5df5-gj7rg   1/1     Running     0          2m38s
model-training-jfnwj            0/1     Completed   0          10h

$ kubectl describe deployment model-serving -n ml-training
...
Replicas:               2 desired | 2 updated | 2 total | 2 available | 0 unavailable
...
    Liveness:   http-get http://:8080/health delay=45s timeout=5s period=10s successThreshold=1 failureThreshold=3
    Readiness:  http-get http://:8080/health delay=15s timeout=5s period=5s successThreshold=1 failureThreshold=3
...
Conditions:
  Type           Status  Reason
  ----           ------  ------
  Available      True    MinimumReplicasAvailable
  Progressing    True    NewReplicaSetAvailable
NewReplicaSet:   model-serving-5bb9c5df5 (2/2 replicas created)

$ kubectl port-forward svc/model-serving 8080:80 -n ml-training
Forwarding from 127.0.0.1:8080 -> 8080
Forwarding from [::1]:8080 -> 8080

$ curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
{"confidence":0.4323,"predicted_class":"cat","probabilities":{"airplane":0.0132,"automobile":0.0001,"bird":0.1595,"cat":0.4323,"deer":0.2177,"dog":0.0161,"frog":0.1472,"horse":0.0072,"ship":0.0066,"truck":0.0002}}
```

## 3. Git workflow followed

- `develop` branched from `main`; all work on feature branches merged via PRs:
  1. `feature/pytorch-model` → model, dataset, training loop, tests (Week 1)
  2. `feature/serving-api` → Flask app, serving tests (Week 1)
  3. `feature/docker-images` → both Dockerfiles, pinned requirements, CI docker job (Week 2)
  4. `feature/k8s-deployment` → all manifests, validation logs (Week 2)
- Conventional Commits used throughout (`feat: add resnet18 variant for 32x32
  inputs`, `test: cover /predict error paths`, `ci: build images after tests`, …).

## 4. Reflection — what was the most challenging part? (write-up)

The most challenging part of this assignment was, by some distance, getting the
handover between the training Job and the serving Deployment to behave sensibly
on Kubernetes. Everything up to that point — the model, the training loop, even
the multi-stage Dockerfiles — produced immediate, visible failures during local 
development. The interaction between the training Job and the serving Deployment's 
shared storage failed differently: silently, and only once deployed. My first 
attempt simply mounted the checkpoint PVC into both workloads and assumed 
ordering would take care of itself; in practice the serving pods came up
before the Job had written `classifier_v1.pt`, the liveness probe saw the app
respond, the readiness probe saw a 503, and I initially misread the resulting
`0/1 Ready` state as a broken probe configuration rather than the system doing
exactly what I had asked. The fix ended up being in the application, not the
YAML: the server now starts without a model, reports 503 from `/health`, and
retries loading the checkpoint on every probe, so the pods join the Service
automatically the moment training finishes. That inverted how I had been
thinking — instead of orchestrating "train, then deploy" from outside, the
serving layer became tolerant of the world it actually starts in.

A second, smaller battle was image size and dependency hygiene. The naive
`pip install torch` pulled the CUDA build and produced a multi-gigabyte image
that took ages to load into minikube. Switching the pinned requirements to the
CPU wheel index, splitting train/serve requirements so the serving image carries
no training extras, and moving dependency installation into a builder stage
brought both images down to something practical and made rebuilds fast because
the dependency layer caches independently of code changes.

Finally, the testing strategy required a design change I had initially been 
reluctant to make: the training loop originally always downloaded CIFAR-10, 
which made CI slow and flaky. Adding a
`fake` dataset option and a `max_batches` cap felt like polluting the config at
first, but it meant the same real code path — logging, early stopping,
checkpointing — runs in seconds in CI and in `run_tests.sh`, which caught two
genuine bugs (a checkpoint field mismatch and a probe path typo) before they
reached the cluster. The lesson I take away is that most of the difficulty in
MLOps is not in any single tool but in the seams between them, and the best
fixes usually make the application more self-sufficient rather than the
orchestration more elaborate.

*(~390 words)*
