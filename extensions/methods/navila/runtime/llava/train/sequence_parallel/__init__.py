def get_pg_manager():
    """Single-worker inference has no sequence-parallel process group."""
    return None
