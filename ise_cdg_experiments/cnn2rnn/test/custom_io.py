from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, Optional, TypeVar


# Because of the Generic class inheritance, 
# # You can use the type "Custom_IO[DATA_TYPE]"
IO_DATA_TYPE = TypeVar("IO_DATA_TYPE")
class Custom_IO(Generic[IO_DATA_TYPE], metaclass=ABCMeta):
    @abstractmethod
    def write(self, data: IO_DATA_TYPE, file_name: str) -> None:
        pass

    @abstractmethod
    def read(self, file_name: str) -> Optional[IO_DATA_TYPE]:
        pass


import json


class JSON_IO(Custom_IO):
    # This class has these features:
    # 1. It is an Adapter of JSON package
    # 2. It puts io in a specific directory
    def __init__(self, results_dir: str):
        self.results_dir = results_dir

    def __get_file_path(self, file_name: str):
        return f"{self.results_dir}/{file_name}"

    def write(self, data: object, file_name: str) -> None:
        with open(self.__get_file_path(file_name), "w") as f:
            json.dump(data, f)

    def read(self, file_name: str) -> Optional[object]:
        with open(self.__get_file_path(file_name)) as f:
            return json.load(f)
