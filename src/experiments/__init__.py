"""Experiments module for running and analyzing experiments."""

_IMPORT_ERROR = None

try:  # pragma: no cover - import side effects only
	from .experiment_runner import ExperimentRunner  # type: ignore
except ModuleNotFoundError as exc:
	# Delay raising the error until ExperimentRunner is accessed explicitly.
	ExperimentRunner = None  # type: ignore
	_IMPORT_ERROR = exc


def __getattr__(name: str):  # pragma: no cover - simple accessor guard
	if name == "ExperimentRunner":
		if _IMPORT_ERROR is not None:
			raise ModuleNotFoundError(
				"ExperimentRunner requires optional plotting dependencies (matplotlib). "
				"Install them or import the class from src.experiments.experiment_runner "
				"after installing matplotlib."
			) from _IMPORT_ERROR
	raise AttributeError(name)


__all__ = ["ExperimentRunner"]