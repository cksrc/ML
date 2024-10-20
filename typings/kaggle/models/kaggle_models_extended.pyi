"""
This type stub file was generated by pyright.
"""

class Competition:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


class SubmitResult:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


class Submission:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class LeaderboardEntry:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


class Dataset:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


class Model:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


class Metadata:
    def __init__(self, init_info) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class DatasetVersion:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class File:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    
    @staticmethod
    def get_size(size, precision=...): # -> str:
        ...
    


class Tag:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


class DatasetNewVersionResponse:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


class DatasetNewResponse:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


class ListFilesResult:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


class Kernel:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


class KernelPushResponse:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


class ModelNewResponse:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


class ModelDeleteResponse:
    def __init__(self, init_dict) -> None:
        ...
    
    def __repr__(self):
        ...
    


def parse(string): # -> datetime:
    ...

class ResumableUploadResult:
    COMPLETE = ...
    FAILED = ...
    INCOMPLETE = ...
    def __init__(self, result, bytes_uploaded=...) -> None:
        ...
    
    def Complete(): # -> ResumableUploadResult:
        ...
    
    def Failed(): # -> ResumableUploadResult:
        ...
    
    def Incomplete(bytes_uploaded=...): # -> ResumableUploadResult:
        ...
    

