"""Logic tests against hand-written `docker inspect` fragments. No Docker or GPU needed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gpu_reload_check as g

HOST = ["/dev/nvidia-modeset", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia0", "/dev/nvidia1", "/dev/nvidiactl"]


def container(devices=(), requests=None, runtime="runc", env=()):
    return {"Name": "/c", "Config": {"Env": list(env)},
            "HostConfig": {"Runtime": runtime, "DeviceRequests": requests,
                           "Devices": [{"PathOnHost": d, "PathInContainer": d} for d in devices]}}


GPU_REQUEST = [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}]


class WantsGpu(unittest.TestCase):
    def test_compose_deploy_resources_request(self):
        self.assertTrue(g.wants_gpu(container(requests=GPU_REQUEST)))

    def test_capability_only_request(self):
        self.assertTrue(g.wants_gpu(container(requests=[{"Driver": "", "Capabilities": [["gpu", "compute"]]}])))

    def test_nvidia_runtime(self):
        self.assertTrue(g.wants_gpu(container(runtime="nvidia")))

    def test_explicit_devices_only(self):
        self.assertTrue(g.wants_gpu(container(devices=["/dev/nvidia0", "/dev/nvidiactl"])))

    def test_plain_container_is_ignored(self):
        self.assertFalse(g.wants_gpu(container()))
        self.assertFalse(g.wants_gpu(container(devices=["/dev/dri/renderD128"])))

    def test_env_var_alone_is_not_a_gpu_container(self):
        # many CUDA base images set this; without the nvidia runtime it does nothing
        self.assertFalse(g.wants_gpu(container(env=["NVIDIA_VISIBLE_DEVICES=all"])))


class Classify(unittest.TestCase):
    def test_no_explicit_devices_is_at_risk(self):
        status, missing = g.classify(container(requests=GPU_REQUEST), HOST)
        self.assertEqual(status, "at-risk")
        self.assertIn("/dev/nvidiactl", missing)

    def test_full_device_list_is_protected(self):
        self.assertEqual(g.classify(container(devices=HOST, requests=GPU_REQUEST), HOST), ("protected", []))

    def test_one_card_of_two_is_enough(self):
        devs = ["/dev/nvidia1", "/dev/nvidiactl", "/dev/nvidia-uvm"]
        self.assertEqual(g.classify(container(devices=devs), HOST)[0], "protected")

    def test_card_without_control_node_is_partial(self):
        status, missing = g.classify(container(devices=["/dev/nvidia0"]), HOST)
        self.assertEqual(status, "partial")
        self.assertIn("/dev/nvidiactl", missing)
        self.assertIn("/dev/nvidia-uvm", missing)

    def test_control_nodes_without_a_card_is_partial(self):
        self.assertEqual(g.classify(container(devices=["/dev/nvidiactl", "/dev/nvidia-uvm"]), HOST)[0], "partial")


class Snippet(unittest.TestCase):
    def test_snippet_is_valid_compose_fragment(self):
        out = g.snippet(["/dev/nvidia0", "/dev/nvidiactl"])
        self.assertEqual(out, "    devices:\n      - /dev/nvidia0:/dev/nvidia0\n      - /dev/nvidiactl:/dev/nvidiactl")


if __name__ == "__main__":
    unittest.main()
