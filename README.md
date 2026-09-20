# docker-gpu-reload-check

Find the Docker containers that will **silently lose their NVIDIA GPU** the next time anything runs
`systemctl daemon-reload`, and print the fix.

## The problem

Every GPU container on my server broke in the same minute: Jellyfin transcodes, a photo-library ML service and
an LLM server. Nothing logged an error. On the host `nvidia-smi` was fine. Inside the containers:

```
Failed to initialize NVML: Unknown Error
cu->cuInit(0) failed -> CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
```

The trigger was an unrelated `systemctl daemon-reload` (I had installed a timer). With Docker's **systemd cgroup
driver on cgroup v2**, the NVIDIA container runtime grants GPU device access behind systemd's back. On a reload,
systemd re-applies the device rules it knows about and the GPU access is gone. Processes keep their open
handles, so they only fail on the next CUDA initialisation, which is why it looks random. Package upgrades
trigger reloads too, so this can hit days after the change that "caused" it.

Full write-up: <https://aaronsawit.com/writing/docker-containers-lose-gpu-after-daemon-reload/>

## The fix

List the device nodes explicitly. Docker registers those with systemd, so they survive a reload:

```yaml
services:
  jellyfin:
    devices:
      - /dev/nvidia0:/dev/nvidia0
      - /dev/nvidiactl:/dev/nvidiactl
      - /dev/nvidia-uvm:/dev/nvidia-uvm
      - /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools
      - /dev/nvidia-modeset:/dev/nvidia-modeset
```

Keep your existing GPU request (`deploy.resources`, `runtime: nvidia` or `--gpus`). This is in addition to it.

## The tool

```bash
curl -O https://raw.githubusercontent.com/aaronsawit/docker-gpu-reload-check/main/gpu_reload_check.py
python3 gpu_reload_check.py
```

```
docker cgroup driver and version: systemd 2

container   GPU now  after daemon-reload
jellyfin    ok       survives
immich_ml   ok       AT RISK (no explicit devices)
llm-server  BROKEN   AT RISK (no explicit devices)
            -> Failed to initialize NVML: Unknown Error: recreate the container (docker compose up -d --force-recreate)

Add this to each at-risk service in its compose file, then recreate it:

    devices:
      - /dev/nvidia-modeset:/dev/nvidia-modeset
      - /dev/nvidia-uvm:/dev/nvidia-uvm
      ...
```

| Command | Does |
| --- | --- |
| `gpu_reload_check.py` | every running GPU container: does the GPU work now, and will it survive a reload |
| `gpu_reload_check.py --snippet` | the `devices:` block for the GPUs on this host, ready to paste |
| `gpu_reload_check.py --json` | the same report as JSON; exit code 1 if any container is at risk (for monitoring) |

It is **read-only**. It runs `docker ps`, `docker inspect` and `docker exec <container> nvidia-smi -L`. It never
reloads systemd and never restarts a container. Python 3.9+, no dependencies.

## Prove the fix yourself

```bash
docker exec jellyfin nvidia-smi -L     # lists the GPUs
sudo systemctl daemon-reload
docker exec jellyfin nvidia-smi -L     # still lists the GPUs
```

Test a fix by repeating the trigger, not by checking that the symptom went away.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The classification logic is tested against hand-written `docker inspect` fragments, so no Docker or GPU is needed.
The tool itself was run against a real two-GPU host.

MIT licence.
