"""
This type stub file was generated by pyright.
"""

import requests
from typing import Optional

GCS_HASH_HEADER = ...
COMPUTE_HASH_CHUNK_SIZE = ...
logger = ...
def get_md5_checksum_from_response(response: requests.Response) -> Optional[str]:
    ...

def update_hash_from_file(hash_object, out_file: str) -> None:
    ...

def to_b64_digest(hash_object) -> str:
    ...

