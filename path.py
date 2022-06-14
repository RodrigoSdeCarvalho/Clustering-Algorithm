import os

class Path:
    def __init__(self) -> None:
        self._base_path = os.path.join(os.getcwd())
        
    @property
    def base_path(self):
        return self._base_path
    
    def dataset_path(self, file_name:str) -> str:
        """Returns the path of the dataset, which must be in the data folder, given the file's name."""
        dataset_path = f'{self._base_path}/data/{file_name}'
        dataset_path = os.path.join(dataset_path)
        
        return dataset_path

    def report_path(self) -> str:
        """Returns the reports' folder path."""
        report_path = f'{self._base_path}/reports/'
        report_path = os.path.join(report_path)
        
        return report_path
