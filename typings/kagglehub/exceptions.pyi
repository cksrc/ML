"""
This type stub file was generated by pyright.
"""

import requests
from typing import Any, Optional
from kagglehub.handle import ResourceHandle

class CredentialError(Exception):
    ...


class KaggleEnvironmentError(Exception):
    ...


class ColabEnvironmentError(Exception):
    ...


class BackendError(Exception):
    def __init__(self, message: str, error_code: Optional[int] = ...) -> None:
        ...
    


class NotFoundError(Exception):
    ...


class DataCorruptionError(Exception):
    ...


class KaggleApiHTTPError(requests.HTTPError):
    def __init__(self, message: str, response: Optional[requests.Response] = ...) -> None:
        ...
    


class ColabHTTPError(requests.HTTPError):
    def __init__(self, message: str, response: Optional[requests.Response] = ...) -> None:
        ...
    


class UnauthenticatedError(Exception):
    """Exception raised for errors in the authentication process."""
    def __init__(self, message: str = ...) -> None:
        ...
    


def kaggle_api_raise_for_status(response: requests.Response, resource_handle: Optional[ResourceHandle] = ...) -> None:
    """
    Wrapper around `response.raise_for_status()` that provides nicer error messages
    See: https://requests.readthedocs.io/en/latest/api/#requests.Response.raise_for_status
    """
    ...

def colab_raise_for_status(response: requests.Response, resource_handle: Optional[ResourceHandle] = ...) -> None:
    """
    Wrapper around `response.raise_for_status()` that provides nicer error messages
    See: https://requests.readthedocs.io/en/latest/api/#requests.Response.raise_for_status
    """
    ...

def process_post_response(response: dict[str, Any]) -> None:
    """
    Postprocesses the API response to check for errors.
    """
    ...

