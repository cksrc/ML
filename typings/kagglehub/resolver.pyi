"""
This type stub file was generated by pyright.
"""

import abc
from typing import Generic, Optional, TypeVar

T = TypeVar("T")
class Resolver(Generic[T]):
    """Resolver base class: all resolvers inherit from this class."""
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def __call__(self, handle: T, path: Optional[str], *, force_download: Optional[bool] = ...) -> str:
        """Resolves a handle into a path with the requested model files.

        Args:
            handle: (string) the model handle to resolve.
            path: (string) Optional path to a file within the model bundle.
            force_download: (bool) Optional flag to force download a model, even if it's cached.


        Returns:
            A string representing the path
        """
        ...
    
    @abc.abstractmethod
    def is_supported(self, handle: T, path: Optional[str]) -> bool:
        """Returns whether the current environment supports this handle/path."""
        ...
    


