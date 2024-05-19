import operator
import threading
from os import PathLike
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ThreadSafeFile:
    """
    Share file objects (e.g., raw binary, buffered binary, and text) between
    threads by storing the file object in thread-local storage (TLS).
    """

    def __init__(
        self,
        path: str | PathLike,
        open: Callable[[str | PathLike], T] = operator.methodcaller("open"),
        close: Callable[[T], None] = operator.methodcaller("close"),
    ) -> None:
        self.path = path

        self.open = open

        self.close = close

        self.storage = threading.local()

    def __del__(self) -> None:
        if hasattr(self.storage, "file"):
            self.close(self.storage.file)

            del self.storage.file

    def __getattr__(self, name: str) -> object:
        return getattr(self.file, name)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()

        if "storage" in state:
            del state["storage"]

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state

        self.storage = threading.local()

    @property
    def file(self) -> T:
        if not hasattr(self.storage, "file"):
            self.storage.file = self.open(self.path)

        return self.storage.file
