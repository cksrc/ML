"""
This type stub file was generated by pyright.
"""

from typing import Optional
from kagglehub.handle import CompetitionHandle, DatasetHandle, ModelHandle
from kagglehub.resolver import Resolver

DATASET_CURRENT_VERSION_FIELD = ...
MODEL_INSTANCE_VERSION_FIELD = ...
MAX_NUM_FILES_DIRECT_DOWNLOAD = ...
logger = ...
class CompetitionHttpResolver(Resolver[CompetitionHandle]):
    def is_supported(self, *_, **__) -> bool:
        ...
    
    def __call__(self, h: CompetitionHandle, path: Optional[str] = ..., *, force_download: Optional[bool] = ...) -> str:
        ...
    


class DatasetHttpResolver(Resolver[DatasetHandle]):
    def is_supported(self, *_, **__) -> bool:
        ...
    
    def __call__(self, h: DatasetHandle, path: Optional[str] = ..., *, force_download: Optional[bool] = ...) -> str:
        ...
    


class ModelHttpResolver(Resolver[ModelHandle]):
    def is_supported(self, *_, **__) -> bool:
        ...
    
    def __call__(self, h: ModelHandle, path: Optional[str] = ..., *, force_download: Optional[bool] = ...) -> str:
        ...
    


