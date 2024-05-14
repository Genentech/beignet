import operator
import threading
from pathlib import Path
from typing import Any, Callable, TypeVar, Union

T = TypeVar("T")


class ThreadSafeFile:
    def __init__(
        self,
        path: Union[str, Path],
        open_function: Callable[[Union[str, Path]], T] = operator.methodcaller(
            "open"
        ),
        close_function: Callable[[T], None] = operator.methodcaller("close"),
    ) -> None:
        """
        Share file objects (i.e., raw binary files, buffered binary files, and
        text files) between threads by storing the file object in thread-local
        storage (TLS).
        """
        self._local = threading.local()

        self._path = path
        self._open_function = open_function
        self._close_function = close_function

    def __getattr__(self, name: str) -> Any:
        return getattr(self.file, name)

    @property
    def file(self) -> T:
        if not hasattr(self._local, "file"):
            self._local.file = self._open_function(self._path)

        return self._local.file

    def __del__(self) -> None:
        if hasattr(self._local, "file"):
            self._close_function(self._local.file)

            del self._local.file

    def __getstate__(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "_local"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state

        self._local = threading.local()
