import operator
import threading
from tempfile import NamedTemporaryFile
from typing import Callable
from unittest.mock import MagicMock

from beignet.io import ThreadSafeFile


class TestThreadSafeFile:
    def test___init__(self):
        with NamedTemporaryFile() as temporary_file:
            thread_safe_file = ThreadSafeFile(temporary_file.name)

            assert thread_safe_file.path == temporary_file.name

            assert isinstance(thread_safe_file.open, Callable)

            assert isinstance(thread_safe_file.close, Callable)

            assert isinstance(thread_safe_file.storage, threading.local)

    def test___del__(self):
        with NamedTemporaryFile() as temporary_file:
            close = MagicMock()

            thread_safe_file = ThreadSafeFile(temporary_file.name, close=close)

            thread_safe_file.storage.file = MagicMock()

            del thread_safe_file

            close.assert_called_once()

    def test___getattr__(self):
        with NamedTemporaryFile() as temporary_file:
            file = MagicMock()

            thread_safe_file = ThreadSafeFile(temporary_file.name)

            thread_safe_file.storage.file = file

            assert thread_safe_file.read == file.read
            assert thread_safe_file.write == file.write

    def test___getstate__(self):
        with NamedTemporaryFile() as temporary_file:
            thread_safe_file = ThreadSafeFile(temporary_file.name)

            state = thread_safe_file.__getstate__()

            assert "path" in state

            assert "open" in state

            assert "close" in state

            assert "storage" not in state

    def test___setstate__(self):
        with NamedTemporaryFile() as temporary_file:
            thread_safe_file = ThreadSafeFile(temporary_file.name)

            thread_safe_file.__setstate__(
                {
                    "path": temporary_file.name,
                    "open": operator.methodcaller("open"),
                    "close": operator.methodcaller("close"),
                },
            )

            assert thread_safe_file.path == temporary_file.name

            assert isinstance(thread_safe_file.open, Callable)

            assert isinstance(thread_safe_file.close, Callable)

            assert isinstance(thread_safe_file.storage, threading.local)

    def test_file(self):
        with NamedTemporaryFile() as temporary_file:
            open = MagicMock(return_value=MagicMock())

            thread_safe_file = ThreadSafeFile(
                temporary_file.name,
                open=open,
            )

            file = thread_safe_file.file

            open.assert_called_once_with(temporary_file.name)

            assert thread_safe_file.storage.file == file
            assert thread_safe_file.file == file

            assert thread_safe_file.file == file
            assert open.call_count == 1
