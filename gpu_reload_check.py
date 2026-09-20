#!/usr/bin/env python3
"""Find Docker containers that will silently lose their NVIDIA GPU on the next `systemctl daemon-reload`.

With the systemd cgroup driver on cgroup v2, the NVIDIA container runtime grants GPU device access
behind systemd's back. Any daemon-reload (installing a timer, a package upgrade) makes systemd
re-apply the device rules it knows about, and the GPU access is gone. The container keeps running
and nothing logs an error until the next CUDA call fails with CUDA_ERROR_NO_DEVICE or
"Failed to initialize NVML: Unknown Error".

Containers that list the /dev/nvidia* nodes explicitly under `devices:` are safe, because Docker
registers those with systemd.

    gpu_reload_check.py              report every running GPU container: works now? survives a reload?
    gpu_reload_check.py --snippet    print the compose `devices:` block for this host's GPUs
    gpu_reload_check.py --json       machine-readable report; exit code 1 if any container is at risk

Read-only: it runs `docker ps`, `docker inspect` and `docker exec <c> nvidia-smi -L`. It never
reloads systemd or restarts anything.
"""
import argparse
import glob
import json
import subprocess
import sys

NVIDIA_NODES = "/dev/nvidia*"


def run(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, "", str(e)


def host_gpu_nodes():
    """Character devices a container needs. /dev/nvidia-caps is a directory and is skipped."""
    return sorted(p for p in glob.glob(NVIDIA_NODES) if not p.endswith("-caps"))


def wants_gpu(inspect):
    """True if the container asked for a GPU by any of the usual routes."""
    host = inspect.get("HostConfig") or {}
    for req in host.get("DeviceRequests") or []:
        caps = [c for group in (req.get("Capabilities") or []) for c in group]
        if req.get("Driver") == "nvidia" or "gpu" in caps:
            return True
    if host.get("Runtime") == "nvidia":
        return True
    env = (inspect.get("Config") or {}).get("Env") or []
    if any(e.startswith("NVIDIA_VISIBLE_DEVICES=") and e.split("=", 1)[1] not in ("", "void", "none") for e in env):
        return host.get("Runtime") == "nvidia" or bool(explicit_nodes(inspect))
    return bool(explicit_nodes(inspect))


def explicit_nodes(inspect):
    devices = (inspect.get("HostConfig") or {}).get("Devices") or []
    return sorted(d.get("PathOnHost", "") for d in devices if d.get("PathOnHost", "").startswith("/dev/nvidia"))


def classify(inspect, host_nodes):
    """Return (status, missing_nodes). status is 'protected', 'partial' or 'at-risk'."""
    have = set(explicit_nodes(inspect))
    # nvidiactl and nvidia-uvm plus at least one card are the minimum that keeps CUDA alive
    needed = [n for n in host_nodes if not n.endswith(("-modeset", "-uvm-tools"))]
    missing = [n for n in needed if n not in have]
    if not have:
        return "at-risk", needed
    cards = [n for n in needed if n[len("/dev/nvidia"):].isdigit()]
    core_missing = [n for n in missing if n not in cards]
    if core_missing or (cards and not any(c in have for c in cards)):
        return "partial", missing
    return "protected", []


def gpu_works(name):
    code, out, err = run(["docker", "exec", name, "nvidia-smi", "-L"])
    if code == 0 and "GPU" in out:
        return True, f"{len(out.splitlines())} GPU(s) visible"
    msg = (out or err).splitlines()[0] if (out or err) else "nvidia-smi failed"
    if "executable file not found" in msg or "not found" in msg:
        return None, "no nvidia-smi in the image, cannot test"
    return False, msg


def snippet(host_nodes):
    lines = ["    devices:"] + [f"      - {n}:{n}" for n in host_nodes]
    return "\n".join(lines)


def report():
    code, out, err = run(["docker", "ps", "--format", "{{.Names}}"])
    if code:
        sys.exit(f"cannot run docker: {err or out}")
    names = out.split()
    if not names:
        return [], host_gpu_nodes()
    code, out, err = run(["docker", "inspect"] + names, timeout=60)
    if code:
        sys.exit(f"docker inspect failed: {err}")
    host_nodes, rows = host_gpu_nodes(), []
    for ins in json.loads(out):
        if not wants_gpu(ins):
            continue
        name = ins["Name"].lstrip("/")
        status, missing = classify(ins, host_nodes)
        works, detail = gpu_works(name)
        rows.append({"container": name, "gpu_works_now": works, "detail": detail,
                     "reload": status, "missing_devices": missing})
    return rows, host_nodes


def cgroup_context():
    _, out, _ = run(["docker", "info", "--format", "{{.CgroupDriver}} {{.CgroupVersion}}"])
    return out or "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snippet", action="store_true", help="print the compose devices: block for this host")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.snippet:
        nodes = host_gpu_nodes()
        if not nodes:
            sys.exit("no /dev/nvidia* device nodes on this host")
        print(snippet(nodes))
        return 0

    rows, host_nodes = report()
    at_risk = [r for r in rows if r["reload"] != "protected"]
    if a.json:
        print(json.dumps({"cgroup": cgroup_context(), "host_devices": host_nodes, "containers": rows}, indent=2))
        return 1 if at_risk else 0

    driver = cgroup_context()
    print(f"docker cgroup driver and version: {driver}")
    if not driver.startswith("systemd 2"):
        print("  (the reload problem is specific to the systemd driver on cgroup v2; this host may not be affected)")
    if not rows:
        print("no running containers use an NVIDIA GPU")
        return 0
    width = max(len(r["container"]) for r in rows)
    print(f"\n{'container':<{width}}  {'GPU now':<8} after daemon-reload")
    for r in rows:
        now = {True: "ok", False: "BROKEN", None: "?"}[r["gpu_works_now"]]
        after = {"protected": "survives", "partial": "AT RISK (device list incomplete)",
                 "at-risk": "AT RISK (no explicit devices)"}[r["reload"]]
        print(f"{r['container']:<{width}}  {now:<8} {after}")
        if r["gpu_works_now"] is False:
            print(f"{'':<{width}}  -> {r['detail']}: recreate the container (docker compose up -d --force-recreate)")
    if at_risk:
        print("\nAdd this to each at-risk service in its compose file, then recreate it:\n")
        print(snippet(host_nodes))
        print("\nKeep your existing GPU request (deploy.resources / runtime: nvidia / --gpus). This is in addition to it.")
    return 1 if at_risk else 0


if __name__ == "__main__":
    sys.exit(main())
