"""Tests for custom exception hierarchy."""

from scoring.exceptions import (
    CaseNotFoundError,
    ConfigurationError,
    FeatureRetrievalError,
    ModelNotLoadedError,
    ModelScoringError,
    ScoringError,
)


class TestExceptionHierarchy:
    """Verify exception inheritance and attributes."""

    def test_all_inherit_from_scoring_error(self):
        for exc_cls in (
            ModelNotLoadedError,
            ModelScoringError,
            FeatureRetrievalError,
            CaseNotFoundError,
            ConfigurationError,
        ):
            assert issubclass(exc_cls, ScoringError)

    def test_scoring_error_inherits_from_exception(self):
        assert issubclass(ScoringError, Exception)

    def test_details_default_to_empty_dict(self):
        exc = ScoringError("test")
        assert exc.details == {}

    def test_details_preserved(self):
        exc = FeatureRetrievalError("oops", details={"user_id": "abc"})
        assert exc.details == {"user_id": "abc"}
        assert str(exc) == "oops"

    def test_model_not_loaded_error(self):
        exc = ModelNotLoadedError("model missing")
        assert isinstance(exc, ScoringError)
        assert str(exc) == "model missing"
