import kagglehub
import kaggle

class KaggleClient:
    def __init__(self):
        self.api = kaggle.api

    def dataset_download(self, dataset_name: str) -> str:
        # Download dataset
        path = kagglehub.dataset_download(dataset_name) # type: ignore
        return path # type: ignore

# Dependency to inject the Kaggle client
def get_kaggle_client() -> KaggleClient:
    return KaggleClient()

