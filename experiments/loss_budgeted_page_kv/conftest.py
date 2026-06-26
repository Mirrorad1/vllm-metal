# SPDX-License-Identifier: Apache-2.0
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("VLLM_METAL_BUILD_FROM_SOURCE", "1")
