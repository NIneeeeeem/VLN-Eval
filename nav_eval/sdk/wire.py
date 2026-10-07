"""Lossless framed JSON + raw tensor buffers, without a numerical dependency."""
import base64
import json
import struct

from nav_eval.contracts import ContractError
from nav_eval.tensorcode import tensor_nbytes, validate_tensor

CONTENT_TYPE = "application/vnd.nav-eval.tensor-stream"
MAX_BINARY_BYTES = 64 * 1024 * 1024


def encode(value):
    buffers = []
    def visit(item):
        if isinstance(item, dict):
            if "__nav_buffer__" in item:
                raise ContractError("reserved binary framing key")
            if "tensor_b64" in item:
                raw = validate_tensor(item)
                index = len(buffers)
                buffers.append(raw)
                return {"__nav_buffer__": index, "dtype": item["dtype"],
                        "shape": item["shape"], "byte_order": item["byte_order"]}
            return {k: visit(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(v) for v in item]
        return item
    tree = visit(value)
    header = json.dumps({"value": tree, "lengths": [len(b) for b in buffers]}, allow_nan=False).encode()
    result = struct.pack("!I", len(header)) + header + b"".join(buffers)
    if len(result) > MAX_BINARY_BYTES:
        raise ContractError("binary message exceeds size limit")
    return result


def decode(body):
    if not 4 <= len(body) <= MAX_BINARY_BYTES:
        raise ContractError("invalid binary message length")
    size = struct.unpack("!I", body[:4])[0]
    if size > len(body) - 4:
        raise ContractError("truncated binary header")
    def reject_constant(value):
        raise ContractError(f"invalid JSON number: {value}")
    try:
        header = json.loads(body[4:4 + size], parse_constant=reject_constant)
    except (ValueError, UnicodeError) as error:
        raise ContractError("invalid binary JSON header") from error
    if not isinstance(header, dict) or set(header) != {"value", "lengths"} or not isinstance(header["lengths"], list):
        raise ContractError("invalid binary header")
    offset, buffers = 4 + size, []
    for length in header["lengths"]:
        if type(length) is not int or length < 0 or offset + length > len(body):
            raise ContractError("invalid binary buffer length")
        buffers.append(body[offset:offset + length])
        offset += length
    if offset != len(body):
        raise ContractError("unexpected trailing binary bytes")
    used = set()
    def visit(item):
        if isinstance(item, dict):
            if "__nav_buffer__" in item:
                if set(item) != {"__nav_buffer__", "shape", "dtype", "byte_order"}:
                    raise ContractError("invalid binary tensor descriptor")
                index = item["__nav_buffer__"]
                if type(index) is not int or not 0 <= index < len(buffers) or index in used:
                    raise ContractError("invalid binary buffer index")
                used.add(index)
                payload = {k: v for k, v in item.items() if k != "__nav_buffer__"}
                if len(buffers[index]) != tensor_nbytes(payload):
                    raise ContractError("tensor buffer length does not match shape and dtype")
                payload["tensor_b64"] = base64.b64encode(buffers[index]).decode("ascii")
                return payload
            return {k: visit(v) for k, v in item.items()}
        if isinstance(item, list):
            return [visit(v) for v in item]
        return item
    result = visit(header["value"])
    if len(used) != len(buffers):
        raise ContractError("unreferenced binary buffer")
    return result
