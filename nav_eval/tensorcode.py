"""Lossless tensor envelope for the development JSON transport.

The wire contract stays versioned JSON; large sensor payloads are base64 of the
raw buffer with explicit dtype/shape/endianness. No compression or resampling:
if an image must be resized that belongs to an explicit, recorded transform.
"""
from __future__ import annotations

import base64
from typing import Any

# The wire is always little-endian, regardless of array or host byte order.
_ENDIAN = "<"

_DTYPES = {
    "uint8": 1,
    "float32": 4,
    "int32": 4,
}


def encode_array(array) -> dict[str, Any]:
    """numpy array -> {"tensor_b64", "dtype", "shape", "byte_order"}."""
    import numpy as np

    if not isinstance(array, np.ndarray):
        raise ValueError("tensor envelope expects a numpy array")
    if array.dtype.name not in _DTYPES:
        raise ValueError(f"unsupported tensor dtype: {array.dtype.name}")
    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder(_ENDIAN))
    return {
        "tensor_b64": base64.b64encode(array.tobytes()).decode("ascii"),
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "byte_order": _ENDIAN,
    }


def decode_tensor(payload: dict[str, Any]):
    """Inverse of encode_array; validates the envelope before materialising."""
    import numpy as np

    raw = validate_tensor(payload)
    return np.frombuffer(raw, dtype=np.dtype(payload["byte_order"] + {"uint8": "u1", "float32": "f4", "int32": "i4"}[payload["dtype"]])).reshape(payload["shape"])


def tensor_nbytes(payload):
    """Validate tensor metadata and return the required raw buffer length."""
    if payload["byte_order"] != _ENDIAN:
        raise ValueError("tensor envelope byte order mismatch")
    dtype = payload["dtype"]
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    shape = payload["shape"]
    if not isinstance(shape, list) or not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("invalid tensor shape")
    expected = _DTYPES[dtype]
    for dim in shape:
        expected *= dim
    return expected


def validate_tensor(payload):
    """Validate buffers without numpy; used by the public SDK and binary transport."""
    if not isinstance(payload, dict) or set(payload) != {"tensor_b64", "dtype", "shape", "byte_order"}:
        raise ValueError("invalid tensor envelope keys")
    expected = tensor_nbytes(payload)
    raw = base64.b64decode(payload["tensor_b64"], validate=True)
    if len(raw) != expected:
        raise ValueError("tensor buffer length does not match shape and dtype")
    return raw
