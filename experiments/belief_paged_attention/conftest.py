# SPDX-License-Identifier: Apache-2.0
"""Make the experiment's sibling modules importable when running pytest here."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
# Kernel tests build from source (the checkout ships no prebuilt .metallib).
os.environ.setdefault("VLLM_METAL_BUILD_FROM_SOURCE", "1")
