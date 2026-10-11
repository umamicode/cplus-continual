#!/usr/bin/env python
"""Fail-fast Cloud TPU/JAX environment check."""

from __future__ import annotations

import jax


def main() -> None:
    devices = jax.devices()
    print("JAX version:", jax.__version__)
    print("Process:", jax.process_index(), "/", jax.process_count())
    print("Local devices:", jax.local_device_count())
    print("Global devices:", jax.device_count())
    print("Devices:", devices)
    if not devices or any(device.platform != "tpu" for device in devices):
        raise SystemExit("Expected TPU devices, but JAX is not using the TPU backend.")
    value = jax.numpy.arange(128, dtype=jax.numpy.bfloat16).reshape(8, 16)
    result = value @ value.T
    result.block_until_ready()
    print("TPU matrix smoke: OK", result.shape, result.dtype)


if __name__ == "__main__":
    main()
