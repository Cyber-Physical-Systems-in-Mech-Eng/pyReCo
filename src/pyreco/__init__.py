"""pyReCo: a Python reservoir computing library."""

try:
    from importlib.metadata import PackageNotFoundError, version

    # the version is not stored in the code: it is derived from the git tag of the release
    # when the package is built (hatch-vcs, see pyproject.toml)
    __version__ = version("pyreco")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "unknown"

#
# from . import layers
# from .layers import InputLayer, ReservoirLayer, ReadoutLayer
#
# from . import custom_models
# from .custom_models import RC