"""Comprehensive plotting module for audit report generation.

This module provides production-quality, deterministic visualizations for
ML model auditing including SHAP explanations, fairness analysis, performance
metrics, and drift detection suitable for regulatory reporting.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any


# Guard against GUI backends in headless environments
def _ensure_headless_backend():
    """Ensure matplotlib uses a headless backend for CI/tests."""
    # Respect an explicit user setting if provided
    if os.environ.get("MPLBACKEND"):
        return
    # On macOS, the default Cocoa backend will crash in non-GUI contexts
    if sys.platform == "darwin":
        import matplotlib

        matplotlib.use("Agg", force=True)
    # On Linux CI without DISPLAY, use Agg
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        import matplotlib

        matplotlib.use("Agg", force=True)


_ensure_headless_backend()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.figure import Figure

# Optional: seaborn for enhanced styling (graceful fallback if not available)
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Professional color palette for regulatory reports
PROFESSIONAL_COLORS = {
    "primary": "#2E3440",  # Dark blue-grey
    "secondary": "#5E81AC",  # Light blue
    "accent": "#88C0D0",  # Cyan
    "warning": "#EBCB8B",  # Yellow
    "error": "#BF616A",  # Red
    "success": "#A3BE8C",  # Green
    "neutral": "#D8DEE9",  # Light grey
    "background": "#ECEFF4",  # Very light grey
}

# Color palette for fairness analysis (colorblind-friendly)
FAIRNESS_COLORS = ["#2E3440", "#5E81AC", "#A3BE8C", "#EBCB8B", "#BF616A", "#B48EAD"]


class AuditPlotter:
    """Professional plotting class for audit report generation."""

    def __init__(self, style: str = "professional", figure_size: tuple[int, int] = (10, 6)):
        """Initialize audit plotter with professional styling.

        Args:
            style: Visual style theme ("professional", "minimal")
            figure_size: Default figure size in inches

        """
        self.style = style
        self.figure_size = figure_size

        # Configure matplotlib for deterministic, high-quality output
        self._configure_matplotlib()

        logger.debug(f"Initialized AuditPlotter with style: {style}")

    def _configure_matplotlib(self) -> None:
        """Configure matplotlib for professional, deterministic plotting."""
        # Set style and parameters
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        rcParams.update(
            {
                # Figure settings
                "figure.figsize": self.figure_size,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.1,
                "savefig.facecolor": "white",
                "savefig.edgecolor": "none",
                # Font settings for professional appearance
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                # Color and style settings
                "axes.facecolor": "white",
                "axes.edgecolor": PROFESSIONAL_COLORS["primary"],
                "axes.linewidth": 0.8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "grid.color": PROFESSIONAL_COLORS["neutral"],
                "grid.linewidth": 0.5,
                "grid.alpha": 0.7,
                # Ensure deterministic behavior
                "figure.max_open_warning": 0,
            },
        )

        # Set color cycle
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=FAIRNESS_COLORS)

    def create_shap_global_importance(
        self,
        feature_importance: dict[str, float],
        title: str = "Global Feature Importance (SHAP)",
        max_features: int = 15,
    ) -> Figure:
        """Create SHAP global feature importance plot.

        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            title: Plot title
            max_features: Maximum number of features to display

        Returns:
            Matplotlib figure object

        """
        logger.debug("Creating SHAP global importance plot")

        if not feature_importance:
            return self._create_empty_plot("No feature importance data available")

        # Sort features by absolute importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:max_features]

        features, importance = zip(*sorted_features, strict=False)

        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Color bars based on positive/negative importance
        colors = [PROFESSIONAL_COLORS["success"] if imp >= 0 else PROFESSIONAL_COLORS["error"] for imp in importance]

        bars = ax.barh(range(len(features)), importance, color=colors, alpha=0.8)

        # Customize plot
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([self._format_feature_name(f) for f in features])
        ax.set_xlabel("SHAP Value (Impact on Model Output)")
        ax.set_title(title, fontweight="bold", pad=20)

        # Add value labels on bars
        for _i, (bar, imp) in enumerate(zip(bars, importance, strict=False)):
            ax.text(
                imp + (max(importance) * 0.01 if imp >= 0 else min(importance) * 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{imp:.3f}",
                ha="left" if imp >= 0 else "right",
                va="center",
                fontsize=8,
            )

        # Add vertical line at x=0
        ax.axvline(x=0, color=PROFESSIONAL_COLORS["primary"], linestyle="-", alpha=0.8)

        plt.tight_layout()
        return fig

    def create_shap_summary_plot(
        self,
        shap_values: np.ndarray | None,
        feature_names: list[str],
        title: str = "SHAP Summary Plot",
    ) -> Figure:
        """Create SHAP summary plot showing feature impact distribution.

        Args:
            shap_values: SHAP values array (n_samples, n_features)
            feature_names: List of feature names
            title: Plot title

        Returns:
            Matplotlib figure object

        """
        logger.debug("Creating SHAP summary plot")

        if shap_values is None or len(shap_values) == 0:
            return self._create_empty_plot("No SHAP values available for summary plot")

        # Ensure 2D array
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        # Calculate feature importance (mean absolute SHAP values)
        importance = np.mean(np.abs(shap_values), axis=0)
        sorted_idx = np.argsort(importance)[::-1][:15]  # Top 15 features

        fig, ax = plt.subplots(figsize=self.figure_size)

        # Create violin plots for each feature
        plot_data = []
        plot_labels = []

        for idx in sorted_idx:
            plot_data.append(shap_values[:, idx])
            plot_labels.append(self._format_feature_name(feature_names[idx]))

        # Create violin plot
        positions = range(len(plot_data))
        violin_parts = ax.violinplot(plot_data, positions=positions, showmeans=True, showmedians=True, vert=False)

        # Customize violin plot colors
        for pc in violin_parts["bodies"]:
            pc.set_facecolor(PROFESSIONAL_COLORS["secondary"])
            pc.set_alpha(0.7)

        # Set labels and title
        ax.set_yticks(positions)
        ax.set_yticklabels(plot_labels)
        ax.set_xlabel("SHAP Value")
        ax.set_title(title, fontweight="bold", pad=20)

        # Add vertical line at x=0
        ax.axvline(x=0, color=PROFESSIONAL_COLORS["primary"], linestyle="-", alpha=0.8)

        plt.tight_layout()
        return fig

    def create_shap_waterfall(
        self,
        base_value: float,
        shap_values: np.ndarray,
        feature_values: dict[str, Any],
        feature_names: list[str],
        title: str = "SHAP Waterfall Plot",
        max_features: int = 10,
    ) -> Figure:
        """Create SHAP waterfall plot for individual prediction explanation.

        Args:
            base_value: Model's base/expected value
            shap_values: SHAP values for individual prediction
            feature_values: Actual feature values for this prediction
            feature_names: List of feature names
            title: Plot title
            max_features: Maximum features to show

        Returns:
            Matplotlib figure object

        """
        logger.debug("Creating SHAP waterfall plot")

        if len(shap_values) == 0:
            return self._create_empty_plot("No SHAP values available for waterfall plot")

        # Get top features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[::-1][:max_features]

        # Prepare data
        values = [base_value]
        labels = ["Base Value"]
        colors = [PROFESSIONAL_COLORS["neutral"]]

        running_total = base_value
        for idx in top_indices:
            shap_val = shap_values[idx]
            feature_name = feature_names[idx]
            feature_val = feature_values.get(feature_name, "N/A")

            values.append(shap_val)
            labels.append(f"{self._format_feature_name(feature_name)}\n= {feature_val}")
            colors.append(PROFESSIONAL_COLORS["success"] if shap_val >= 0 else PROFESSIONAL_COLORS["error"])

            running_total += shap_val

        # Add final prediction
        values.append(0)  # No additional value for final bar
        labels.append(f"Prediction\n{running_total:.3f}")
        colors.append(PROFESSIONAL_COLORS["primary"])

        # Create waterfall chart
        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate cumulative positions for waterfall effect
        cumulative = np.cumsum([0] + values[:-1])

        for i, (val, _label, color) in enumerate(zip(values, labels, colors, strict=False)):
            if i == 0:  # Base value
                ax.bar(i, val, color=color, alpha=0.8, width=0.6)
                ax.text(i, val / 2, f"{val:.3f}", ha="center", va="center", fontweight="bold")
            elif i == len(values) - 1:  # Final prediction
                final_height = cumulative[i]
                ax.bar(i, final_height, color=color, alpha=0.8, width=0.6)
                ax.text(i, final_height / 2, f"{final_height:.3f}", ha="center", va="center", fontweight="bold")
            else:  # SHAP contributions
                if val >= 0:
                    ax.bar(i, val, bottom=cumulative[i], color=color, alpha=0.8, width=0.6)
                    ax.text(i, cumulative[i] + val / 2, f"{val:+.3f}", ha="center", va="center", fontweight="bold")
                else:
                    ax.bar(i, abs(val), bottom=cumulative[i] + val, color=color, alpha=0.8, width=0.6)
                    ax.text(i, cumulative[i] + val / 2, f"{val:+.3f}", ha="center", va="center", fontweight="bold")

                # Draw connecting lines
                if i < len(values) - 1:
                    y_start = cumulative[i] + val if val >= 0 else cumulative[i]
                    y_end = cumulative[i + 1] if i < len(values) - 2 else cumulative[i] + val
                    ax.plot([i + 0.3, i + 0.7], [y_start, y_end], "k--", alpha=0.5, linewidth=1)

        # Customize plot
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Model Output")
        ax.set_title(title, fontweight="bold", pad=20)

        plt.tight_layout()
        return fig

    def create_performance_metrics_summary(
        self,
        metrics: dict[str, float | dict[str, Any]],
        title: str = "Model Performance Summary",
    ) -> Figure:
        """Create performance metrics summary visualization.

        Args:
            metrics: Dictionary of performance metrics
            title: Plot title

        Returns:
            Matplotlib figure object

        """
        logger.debug("Creating performance metrics summary")

        if not metrics:
            return self._create_empty_plot("No performance metrics available")

        # Extract simple metrics (float values)
        simple_metrics = {}
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                simple_metrics[name] = value
            elif isinstance(value, dict) and "value" in value:
                simple_metrics[name] = value["value"]
            elif isinstance(value, dict) and name == "classification_report":
                # Extract key metrics from classification report
                if "accuracy" in value:
                    simple_metrics["Accuracy"] = value["accuracy"]
                if "macro_f1" in value:
                    simple_metrics["Macro F1"] = value["macro_f1"]
                if "weighted_f1" in value:
                    simple_metrics["Weighted F1"] = value["weighted_f1"]

        if not simple_metrics:
            return self._create_empty_plot("No displayable performance metrics")

        # Create subplot layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left plot: Bar chart of metrics
        metric_names = list(simple_metrics.keys())
        metric_values = list(simple_metrics.values())

        bars = ax1.bar(range(len(metric_names)), metric_values, color=PROFESSIONAL_COLORS["secondary"], alpha=0.8)

        ax1.set_xticks(range(len(metric_names)))
        ax1.set_xticklabels([self._format_metric_name(name) for name in metric_names], rotation=45, ha="right")
        ax1.set_ylabel("Score")
        ax1.set_title("Performance Metrics", fontweight="bold")
        ax1.set_ylim(0, 1.1)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values, strict=False):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Right plot: Radar chart for key metrics
        self._create_performance_radar(ax2, simple_metrics)

        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.95)
        plt.tight_layout()
        return fig

    def create_fairness_analysis(
        self,
        fairness_results: dict[str, dict[str, Any]],
        title: str = "Fairness Analysis",
    ) -> Figure:
        """Create comprehensive fairness analysis visualization.

        Args:
            fairness_results: Dictionary with fairness metrics by protected attribute
            title: Plot title

        Returns:
            Matplotlib figure object

        """
        logger.debug("Creating fairness analysis plot")

        if not fairness_results:
            return self._create_empty_plot("No fairness analysis data available")

        # Count available attributes and metrics
        n_attributes = len(fairness_results)

        if n_attributes == 0:
            return self._create_empty_plot("No fairness metrics computed")

        # Create subplot layout
        fig = plt.figure(figsize=(15, 6 * n_attributes))

        for i, (attr_name, attr_metrics) in enumerate(fairness_results.items()):
            # Filter out error results
            valid_metrics = {k: v for k, v in attr_metrics.items() if isinstance(v, dict) and "error" not in v}

            if not valid_metrics:
                continue

            ax = fig.add_subplot(n_attributes, 2, 2 * i + 1)
            self._create_fairness_bars(ax, valid_metrics, attr_name)

            ax = fig.add_subplot(n_attributes, 2, 2 * i + 2)
            self._create_bias_indicator(ax, valid_metrics, attr_name)

        plt.suptitle(title, fontsize=16, fontweight="bold", y=0.95)
        plt.tight_layout()
        return fig

    def create_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str] | None = None,
        title: str = "Confusion Matrix",
    ) -> Figure:
        """Create confusion matrix heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for classes
            title: Plot title

        Returns:
            Matplotlib figure object

        """
        logger.debug("Creating confusion matrix")

        try:
            from sklearn.metrics import confusion_matrix
        except ImportError:
            raise ImportError("sklearn not available - install scikit-learn or fix CI environment") from None

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if class_names is None:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create heatmap with custom colormap
        if HAS_SEABORN:
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={"label": "Count"},
                ax=ax,
            )
        else:
            # Fallback to matplotlib imshow if seaborn not available
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.figure.colorbar(im, ax=ax, label="Count")

            # Set ticks and labels
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)

            # Add text annotations
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )

        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=20)

        # Calculate and display accuracy
        accuracy = np.diagonal(cm).sum() / cm.sum()
        ax.text(
            0.02,
            0.98,
            f"Accuracy: {accuracy:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        return fig

    def create_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, title: str = "ROC Curve") -> Figure:
        """Create ROC curve plot.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            title: Plot title

        Returns:
            Matplotlib figure object

        """
        logger.debug("Creating ROC curve")

        try:
            from sklearn.metrics import auc, roc_curve
        except ImportError:
            raise ImportError("sklearn not available - install scikit-learn or fix CI environment") from None

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot ROC curve
        ax.plot(fpr, tpr, color=PROFESSIONAL_COLORS["secondary"], linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")

        # Plot diagonal reference line
        ax.plot(
            [0, 1],
            [0, 1],
            color=PROFESSIONAL_COLORS["neutral"],
            linestyle="--",
            linewidth=1,
            label="Random Classifier",
        )

        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=20)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_feature_distribution(
        self,
        data: pd.DataFrame,
        feature: str,
        target: str | None = None,
        title: str | None = None,
    ) -> Figure:
        """Create feature distribution plot.

        Args:
            data: DataFrame with feature data
            feature: Feature column name
            target: Target column name for stratification
            title: Plot title

        Returns:
            Matplotlib figure object

        """
        logger.debug(f"Creating feature distribution plot for {feature}")

        if feature not in data.columns:
            return self._create_empty_plot(f"Feature '{feature}' not found in data")

        title = title or f"Distribution of {self._format_feature_name(feature)}"

        fig, ax = plt.subplots(figsize=self.figure_size)

        if target and target in data.columns:
            # Stratified by target
            for i, class_val in enumerate(sorted(data[target].unique())):
                subset = data[data[target] == class_val]
                if subset[feature].dtype in ["object", "category"]:
                    # Categorical feature
                    counts = subset[feature].value_counts()
                    ax.bar(
                        [f"{x}_{class_val}" for x in counts.index],
                        counts.values,
                        alpha=0.7,
                        label=f"Target = {class_val}",
                        color=FAIRNESS_COLORS[i % len(FAIRNESS_COLORS)],
                    )
                else:
                    # Numerical feature
                    ax.hist(
                        subset[feature].dropna(),
                        bins=20,
                        alpha=0.7,
                        label=f"Target = {class_val}",
                        color=FAIRNESS_COLORS[i % len(FAIRNESS_COLORS)],
                    )
            ax.legend()
        # Single distribution
        elif data[feature].dtype in ["object", "category"]:
            counts = data[feature].value_counts()
            ax.bar(counts.index, counts.values, color=PROFESSIONAL_COLORS["secondary"], alpha=0.8)
        else:
            ax.hist(data[feature].dropna(), bins=20, color=PROFESSIONAL_COLORS["secondary"], alpha=0.8)

        ax.set_xlabel(self._format_feature_name(feature), fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=20)

        plt.tight_layout()
        return fig

    def _create_performance_radar(self, ax, metrics: dict[str, float]) -> None:
        """Create radar chart for performance metrics."""
        if len(metrics) < 3:
            ax.text(0.5, 0.5, "Not enough metrics\nfor radar chart", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Performance Radar", fontweight="bold")
            return

        # Select up to 6 key metrics for radar chart
        key_metrics = dict(list(metrics.items())[:6])

        angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        values = list(key_metrics.values()) + [list(key_metrics.values())[0]]

        ax = plt.subplot(111, projection="polar")
        ax.plot(angles, values, "o-", linewidth=2, color=PROFESSIONAL_COLORS["secondary"])
        ax.fill(angles, values, alpha=0.25, color=PROFESSIONAL_COLORS["secondary"])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self._format_metric_name(name) for name in key_metrics])
        ax.set_ylim(0, 1)
        ax.set_title("Performance Radar", fontweight="bold", pad=20)

    def _create_fairness_bars(self, ax, metrics: dict[str, dict], attr_name: str) -> None:
        """Create bar chart for fairness metrics."""
        metric_names = []
        fairness_scores = []
        bias_indicators = []

        for name, result in metrics.items():
            metric_names.append(self._format_metric_name(name))

            if "ratio" in result:
                fairness_scores.append(abs(1.0 - result["ratio"]))
            elif "difference" in result:
                fairness_scores.append(abs(result["difference"]))
            else:
                fairness_scores.append(0.0)

            bias_indicators.append(result.get("is_fair", True))

        # Color bars based on bias detection
        colors = [PROFESSIONAL_COLORS["success"] if fair else PROFESSIONAL_COLORS["error"] for fair in bias_indicators]

        bars = ax.bar(range(len(metric_names)), fairness_scores, color=colors, alpha=0.8)

        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha="right")
        ax.set_ylabel("Bias Score")
        ax.set_title(f"Fairness Metrics: {attr_name}", fontweight="bold")

        # Add value labels
        for bar, score in zip(bars, fairness_scores, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    def _create_bias_indicator(self, ax, metrics: dict[str, dict], attr_name: str) -> None:
        """Create bias indicator visualization."""
        bias_counts = {"Fair": 0, "Biased": 0}

        for result in metrics.values():
            if result.get("is_fair", True):
                bias_counts["Fair"] += 1
            else:
                bias_counts["Biased"] += 1

        # Create pie chart
        colors = [PROFESSIONAL_COLORS["success"], PROFESSIONAL_COLORS["error"]]
        wedges, texts, autotexts = ax.pie(
            bias_counts.values(),
            labels=bias_counts.keys(),
            colors=colors,
            autopct="%1.0f%%",
            startangle=90,
        )

        ax.set_title(f"Bias Detection: {attr_name}", fontweight="bold")

    def _create_empty_plot(self, message: str) -> Figure:
        """Create empty plot with informative message."""
        fig, ax = plt.subplots(figsize=self.figure_size)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig

    def _format_feature_name(self, name: str) -> str:
        """Format feature name for display."""
        # Replace underscores with spaces and title case
        formatted = name.replace("_", " ").title()
        # Truncate long names
        if len(formatted) > 20:
            formatted = formatted[:17] + "..."
        return formatted

    def _format_metric_name(self, name: str) -> str:
        """Format metric name for display."""
        # Handle common metric name formatting
        name_map = {
            "auc_roc": "AUC-ROC",
            "f1": "F1 Score",
            "precision": "Precision",
            "recall": "Recall",
            "accuracy": "Accuracy",
            "demographic_parity": "Demographic Parity",
            "equal_opportunity": "Equal Opportunity",
            "equalized_odds": "Equalized Odds",
            "predictive_parity": "Predictive Parity",
        }

        return name_map.get(name.lower(), name.replace("_", " ").title())

    def save_plot(self, fig: Figure, path: Path, format: str = "png") -> Path:
        """Save plot to file with high quality settings.

        Args:
            fig: Matplotlib figure to save
            path: Output file path
            format: Output format ('png', 'pdf', 'svg')

        Returns:
            Path to saved file

        """
        output_path = path.with_suffix(f".{format}")

        fig.savefig(
            output_path,
            format=format,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
            facecolor="white",
            edgecolor="none",
        )

        plt.close(fig)
        logger.info(f"Saved plot to {output_path}")

        return output_path


# Convenience functions for common plot types
def create_shap_plots(
    explanations: dict[str, Any],
    feature_names: list[str],
    style: str = "professional",
) -> dict[str, Figure]:
    """Create comprehensive SHAP visualization suite.

    Args:
        explanations: Dictionary with SHAP explanation results
        feature_names: List of feature names
        style: Visual style theme

    Returns:
        Dictionary mapping plot names to figure objects

    """
    plotter = AuditPlotter(style=style)
    plots = {}

    # Global importance plot
    if "global_importance" in explanations:
        plots["global_importance"] = plotter.create_shap_global_importance(explanations["global_importance"])

    # Summary plot
    if "shap_values" in explanations:
        plots["summary"] = plotter.create_shap_summary_plot(explanations["shap_values"], feature_names)

    # Waterfall plot for first sample
    if (
        all(key in explanations for key in ["shap_values", "base_value", "feature_values"])
        and len(explanations["shap_values"]) > 0
    ):
        plots["waterfall"] = plotter.create_shap_waterfall(
            explanations["base_value"],
            explanations["shap_values"][0],  # First sample
            explanations["feature_values"],
            feature_names,
        )

    return plots


def create_performance_plots(
    performance_results: dict[str, Any],
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
    style: str = "professional",
) -> dict[str, Figure]:
    """Create performance analysis visualization suite.

    Args:
        performance_results: Dictionary with performance metrics
        y_true: True labels (for confusion matrix and ROC)
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        style: Visual style theme

    Returns:
        Dictionary mapping plot names to figure objects

    """
    plotter = AuditPlotter(style=style)
    plots = {}

    # Performance summary
    plots["summary"] = plotter.create_performance_metrics_summary(performance_results)

    # Confusion matrix
    if y_true is not None and y_pred is not None:
        plots["confusion_matrix"] = plotter.create_confusion_matrix(y_true, y_pred)

    # ROC curve
    if y_true is not None and y_proba is not None:
        plots["roc_curve"] = plotter.create_roc_curve(y_true, y_proba)

    return plots


def create_fairness_plots(
    fairness_results: dict[str, dict[str, Any]],
    style: str = "professional",
) -> dict[str, Figure]:
    """Create fairness analysis visualization suite.

    Args:
        fairness_results: Dictionary with fairness analysis results
        style: Visual style theme

    Returns:
        Dictionary mapping plot names to figure objects

    """
    plotter = AuditPlotter(style=style)
    plots = {}

    # Main fairness analysis
    plots["fairness_analysis"] = plotter.create_fairness_analysis(fairness_results)

    return plots


def plot_drift_analysis(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: list[str],
    style: str = "professional",
) -> dict[str, Figure]:
    """Create drift analysis visualization suite.

    Args:
        reference_data: Reference/training dataset
        current_data: Current/production dataset
        features: Features to analyze for drift
        style: Visual style theme

    Returns:
        Dictionary mapping plot names to figure objects

    """
    plotter = AuditPlotter(style=style)
    plots = {}

    # Feature distribution comparisons
    for feature in features[:6]:  # Limit to first 6 features
        if feature in reference_data.columns and feature in current_data.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Reference distribution
            if reference_data[feature].dtype in ["object", "category"]:
                counts = reference_data[feature].value_counts()
                ax1.bar(counts.index, counts.values, color=PROFESSIONAL_COLORS["secondary"], alpha=0.8)
            else:
                ax1.hist(reference_data[feature].dropna(), bins=20, color=PROFESSIONAL_COLORS["secondary"], alpha=0.8)
            ax1.set_title(f"Reference: {plotter._format_feature_name(feature)}")

            # Current distribution
            if current_data[feature].dtype in ["object", "category"]:
                counts = current_data[feature].value_counts()
                ax2.bar(counts.index, counts.values, color=PROFESSIONAL_COLORS["accent"], alpha=0.8)
            else:
                ax2.hist(current_data[feature].dropna(), bins=20, color=PROFESSIONAL_COLORS["accent"], alpha=0.8)
            ax2.set_title(f"Current: {plotter._format_feature_name(feature)}")

            plots[f"drift_{feature}"] = fig

    return plots
