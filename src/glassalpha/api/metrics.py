"""Metric section wrappers with dict + attribute access.

Phase 2: ReadonlyMetrics base class with deep immutability.
"""

from __future__ import annotations

import types
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np


def _freeze_nested(obj: Any) -> Any:
    """Recursively freeze mutable containers.

    Converts:
    - dict → MappingProxyType (nested frozen dicts recursively)
    - list → tuple (recursively)
    - np.ndarray → read-only C-contiguous array
    - Other types → pass through

    Args:
        obj: Object to freeze

    Returns:
        Immutable version of object

    """
    if isinstance(obj, dict):
        # Recursively freeze nested dicts, then wrap in MappingProxyType
        frozen_dict = {k: _freeze_nested(v) for k, v in obj.items()}
        return types.MappingProxyType(frozen_dict)
    if isinstance(obj, list):
        return tuple(_freeze_nested(x) for x in obj)
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        arr.setflags(write=False)
        return arr
    return obj


class ReadonlyMetrics:
    """Base class for immutable metric sections with Mapping + attribute access.

    Provides both dict-style and attribute-style access:
    - Dict-style: result.performance["accuracy"] (raises KeyError if missing)
    - Attribute-style: result.performance.accuracy (raises GlassAlphaError if missing)

    All nested data is recursively frozen to prevent mutation.
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        """Initialize with frozen data.

        Args:
            data: Metric dictionary to wrap

        """
        # Freeze nested dicts recursively (returns MappingProxyType if dict)
        if isinstance(data, types.MappingProxyType):
            # Already frozen
            frozen = data
        else:
            frozen = _freeze_nested(dict(data))
        object.__setattr__(self, "_data", frozen)

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        """Dict-style access: result.performance['accuracy'].

        Raises:
            KeyError: If metric not in result

        """
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over metric names."""
        return iter(self._data)

    def __len__(self) -> int:
        """Number of metrics."""
        return len(self._data)

    def keys(self):
        """View of metric names."""
        return self._data.keys()

    def values(self):
        """View of metric values."""
        return self._data.values()

    def items(self):
        """View of (name, value) pairs."""
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get metric with default.

        Args:
            key: Metric name
            default: Default value if not found

        Returns:
            Metric value or default

        """
        return self._data.get(key, default)

    # Attribute access
    def __getattr__(self, name: str) -> Any:
        """Attribute-style access: result.performance.accuracy.

        Raises GlassAlphaError (not AttributeError) for unknown metrics.
        This provides better error messages with docs links.

        Phase 3 will integrate with metric registry for helpful errors.
        """
        # Phase 3: Will use metric registry for better errors
        # For now, just raise AttributeError with helpful message
        try:
            return self._data[name]
        except KeyError:
            msg = (
                f"Metric '{name}' not available in this result. "
                f"Check {self.__class__.__name__.lower().replace('metrics', '')}.keys() "
                "for available metrics."
            )
            raise AttributeError(msg) from None

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({dict(self._data)})"

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent attribute mutation."""
        msg = f"{self.__class__.__name__} is immutable"
        raise AttributeError(msg)


class PerformanceMetrics(ReadonlyMetrics):
    """Performance metrics with optional plotting.

    Metrics requiring y_proba:
    - roc_auc, pr_auc, brier_score, log_loss

    Accessing these without probabilities will raise a helpful error
    in Phase 5 (when GlassAlphaError is implemented).
    """

    # Metrics requiring y_proba
    _PROBA_REQUIRED = {"roc_auc", "pr_auc", "brier_score", "log_loss"}

    def plot_confusion_matrix(self, ax=None, normalize=False, cmap="Blues", title="Confusion Matrix"):
        """Plot confusion matrix from stored metrics.

        Args:
            ax: Matplotlib axes (optional, creates new figure if None)
            normalize: If True, normalize confusion matrix values
            cmap: Matplotlib colormap name
            title: Plot title

        Returns:
            Matplotlib axes object

        Raises:
            ImportError: If matplotlib not available
            KeyError: If confusion_matrix not in metrics

        Examples:
            >>> fig, ax = plt.subplots()
            >>> result.performance.plot_confusion_matrix(ax=ax)
            >>> plt.show()

        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            msg = "matplotlib required for plotting. Install with: pip install glassalpha[all]"
            raise ImportError(msg) from e

        # Get confusion matrix from stored metrics
        cm = self._data.get("confusion_matrix")
        if cm is None:
            msg = "confusion_matrix not available in performance metrics"
            raise KeyError(msg)

        # Convert to numpy array
        cm_array = np.array(cm)

        # Normalize if requested
        if normalize:
            cm_array = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True)

        # Create figure if ax not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        # Plot heatmap
        im = ax.imshow(cm_array, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        # Set ticks and labels
        n_classes = cm_array.shape[0]
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xlabel="Predicted Label",
            ylabel="True Label",
            title=title,
        )

        # Add text annotations
        fmt = ".2f" if normalize else "d"
        thresh = cm_array.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(
                    j,
                    i,
                    format(cm_array[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm_array[i, j] > thresh else "black",
                )

        ax.set_ylim(n_classes - 0.5, -0.5)  # Flip y-axis for matrix orientation
        return ax

    def plot_roc_curve(self, ax=None, title="ROC Curve"):
        """Plot ROC curve from stored TPR/FPR arrays.

        Note: This requires roc_curve_data to be stored in metrics
        (currently not implemented - will show informative error).

        Args:
            ax: Matplotlib axes (optional, creates new figure if None)
            title: Plot title

        Returns:
            Matplotlib axes object

        Raises:
            ImportError: If matplotlib not available
            NotImplementedError: ROC curve data not yet stored in metrics

        Examples:
            >>> fig, ax = plt.subplots()
            >>> result.performance.plot_roc_curve(ax=ax)
            >>> plt.show()

        """
        msg = (
            "ROC curve plotting requires storing FPR/TPR arrays in metrics. "
            "This will be implemented in Phase 3. "
            "For now, use the PDF report which includes ROC curves."
        )
        raise NotImplementedError(msg)


class FairnessMetrics(ReadonlyMetrics):
    """Fairness metrics with group-level details."""

    def plot_group_metrics(self, metric: str = "selection_rate", ax=None, title=None):
        """Plot fairness metric by protected groups.

        Args:
            metric: Metric to plot ("selection_rate", "tpr", "fpr", "precision", "recall")
            ax: Matplotlib axes (optional, creates new figure if None)
            title: Plot title (auto-generated if None)

        Returns:
            Matplotlib axes object

        Raises:
            ImportError: If matplotlib not available
            ValueError: If no group metrics available

        Examples:
            >>> fig, ax = plt.subplots()
            >>> result.fairness.plot_group_metrics(metric="selection_rate", ax=ax)
            >>> plt.show()

        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            msg = "matplotlib required for plotting. Install with: pip install glassalpha[all]"
            raise ImportError(msg) from e

        # Get group metrics from stored fairness data
        group_metrics = self._data.get("group_metrics")
        if not group_metrics:
            msg = "No group_metrics available. Ensure protected_attributes were provided during audit."
            raise ValueError(msg)

        # Extract groups and values for the requested metric
        groups = []
        values = []
        for attr_name, attr_groups in group_metrics.items():
            if isinstance(attr_groups, dict):
                for group_name, group_values in attr_groups.items():
                    if isinstance(group_values, dict) and metric in group_values:
                        groups.append(f"{attr_name}={group_name}")
                        values.append(group_values[metric])

        if not groups:
            msg = f"Metric '{metric}' not found in group_metrics. Available: {list(group_metrics.keys())}"
            raise ValueError(msg)

        # Create figure if ax not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # Create bar plot
        x_pos = np.arange(len(groups))
        bars = ax.bar(x_pos, values, alpha=0.8, edgecolor="black")

        # Color bars by fairness threshold (0.80 parity threshold)
        if metric in ["selection_rate", "tpr", "fpr"]:
            mean_val = np.mean(values)
            for bar, val in zip(bars, values, strict=False):
                ratio = val / mean_val if mean_val > 0 else 1.0
                if 0.8 <= ratio <= 1.25:  # Within 80% parity
                    bar.set_color("green")
                    bar.set_alpha(0.6)
                else:
                    bar.set_color("red")
                    bar.set_alpha(0.6)

        # Labels and title
        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        if title is None:
            title = f"Fairness: {metric.replace('_', ' ').title()} by Group"
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

        # Add horizontal line at mean
        ax.axhline(y=np.mean(values), color="black", linestyle="--", alpha=0.5, label="Mean")
        ax.legend()

        plt.tight_layout()
        return ax


class CalibrationMetrics(ReadonlyMetrics):
    """Calibration metrics."""

    def plot(self, ax=None, title="Calibration Curve"):
        """Plot calibration curve from stored bin data.

        Note: Currently requires enabling confidence intervals to store bin data.
        This will be simplified in Phase 3.

        Args:
            ax: Matplotlib axes (optional, creates new figure if None)
            title: Plot title

        Returns:
            Matplotlib axes object

        Raises:
            ImportError: If matplotlib not available
            NotImplementedError: Bin data not yet stored by default

        Examples:
            >>> # For now, use the PDF report which includes calibration plots
            >>> result.to_pdf("audit.pdf")

        """
        msg = (
            "Calibration curve plotting requires bin data to be stored in metrics. "
            "This will be implemented in Phase 3. "
            "For now, use the PDF report which includes calibration curves."
        )
        raise NotImplementedError(msg)


class StabilityMetrics(ReadonlyMetrics):
    """Stability test results."""


class ExplanationSummary(ReadonlyMetrics):
    """SHAP explanation summary."""


class RecourseSummary(ReadonlyMetrics):
    """Recourse generation summary."""
