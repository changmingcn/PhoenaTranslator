"""PhoenaTranslator application package.

Importing this package performs no directory creation, network access, task
recovery, worker startup or credential validation.
"""

from .config import AppConfig

__all__ = ["AppConfig"]
