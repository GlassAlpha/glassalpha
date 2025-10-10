# SPDX-License-Identifier: Apache-2.0
"""Contract guard tests - prevent regression of the 4 chronic failures.

These tests validate the exact behaviors that were causing CI thrashing.
They run against the installed package to match what CI tests.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_init_logging_exact() -> None:
    """Guard test: Logger uses exact f-string format (single argument).

    Prevents regression of: logger.info("%s", arg) vs logger.info(f"{arg}")
    CI test expects: assert_called_with("Initialized audit pipeline with profile: tabular_compliance")
    """
    # Import after installation to test the wheel
    import glassalpha.pipeline.audit as audit_mod  # noqa: PLC0415
    from glassalpha.pipeline.audit import AuditPipeline  # noqa: PLC0415

    with patch.object(audit_mod, "logger") as spy:
        AuditPipeline(config=SimpleNamespace(audit_profile="tabular_compliance"))

        # This is the exact call the CI test expects
        spy.info.assert_any_call("Initialized audit pipeline with profile: tabular_compliance")


def test_standard_template_packaged() -> None:
    """Guard test: Template is packaged inside the installed wheel.

    Prevents regression of: TemplateNotFound errors in CI
    Template must be accessible via importlib.resources from installed package.
    """
    from importlib.resources import files  # noqa: PLC0415

    p = files("glassalpha.report.templates").joinpath("standard_audit.html")
    assert p.is_file(), f"Missing template in installed package: {p}"


@pytest.mark.skipif(
    condition=False,  # Enable now - sklearn deps are available
    reason="Requires sklearn dependencies - enable when ready",
)
def test_lr_roundtrip(tmp_path: Path) -> None:
    """Guard test: LogisticRegression save/load is symmetric and works post-load.

    Prevents regression of: save/load asymmetry causing "model is None" after load()
    Must handle renamed columns and return self on load().
    """
    import pandas as pd  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper  # noqa: PLC0415

    X = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]})
    y = [0, 1, 0, 1]

    w = LogisticRegressionWrapper()
    w.model = LogisticRegression()
    w.fit(X, y, random_state=42)

    path = tmp_path / "lr.joblib"
    w.save(path)

    # Load into new wrapper
    w2 = LogisticRegressionWrapper().load(path)
    assert w2.model is not None, "Model should not be None after load()"

    # Renamed columns must still predict (tests _prepare_X robustness)
    X_renamed = X.rename(columns={"a": "A", "b": "B"})
    predictions = w2.predict(X_renamed)
    assert len(predictions) == len(X), "Should predict on renamed columns"


@pytest.mark.skipif(
    condition=False,  # Enable now - manifest dependencies are available
    reason="Requires manifest dependencies - enable when ready",
)
def test_manifest_tracks_model() -> None:
    """Guard test: Manifest tracks the selected model in expected shape.

    Prevents regression of: selected_components["model"] missing or wrong format
    E2E tests expect: selected_components["model"] = {"name": "lightgbm", "type": "model"}
    """
    from glassalpha.utils.manifest import ManifestGenerator  # noqa: PLC0415

    m = ManifestGenerator()
    m.add_component("model", "lightgbm")

    assert "model" in m.manifest.selected_components
    assert m.manifest.selected_components["model"] == {
        "name": "lightgbm",
        "type": "model",
    }, "Manifest component format must match E2E test expectations"


def test_wheel_contains_all_contracts() -> None:
    """Meta-test: Verify all 4 contracts are present in the built wheel.

    This runs the same checks as wheel_smoke.sh but as a pytest test.
    Useful for CI validation that doesn't rely on bash scripts.
    """
    import zipfile  # noqa: PLC0415

    # Find the wheel (should be built by CI)
    dist_path = Path("dist")
    if not dist_path.exists():
        pytest.skip("No dist/ directory - wheel not built")

    wheels = list(dist_path.glob("*.whl"))
    if not wheels:
        pytest.skip("No wheel file found in dist/")

    wheel = wheels[0]

    with zipfile.ZipFile(wheel, "r") as zf:
        files = zf.namelist()

        # Contract 1: Template packaged
        assert "glassalpha/report/templates/standard_audit.html" in files, "Template not packaged in wheel"

        # Contract 2: Logger format
        audit_source = zf.read("glassalpha/pipeline/audit.py").decode("utf-8")
        assert 'logger.info(f"Initialized audit pipeline with profile:' in audit_source, (
            "Logger not using f-string format"
        )

        # Contract 3: Training logic
        assert 'if getattr(self.model, "model", None) is None:' in audit_source, "Simplified training logic missing"
        assert "self.model.fit(X_processed, y_true" in audit_source, "Model fit call missing"

        # Contract 4: Save/load symmetry
        sklearn_source = zf.read("glassalpha/models/tabular/sklearn.py").decode("utf-8")
        assert "return self" in sklearn_source, "Load method doesn't return self"
        assert "self._is_fitted = True" in sklearn_source, "Load doesn't set _is_fitted"
        assert '"n_classes": len(getattr(self.model, "classes_"' in sklearn_source, "Save doesn't include n_classes"
