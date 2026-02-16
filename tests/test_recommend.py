"""Tests for the recommendation engine."""

import numpy as np
import pytest

from muzik.recommend import _cosine_sim, _hybrid_score, _is_zero


class TestCosineSimlarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_sim(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_sim(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert _cosine_sim(a, b) == 0.0


class TestIsZero:
    def test_none(self) -> None:
        assert _is_zero(None) is True

    def test_zero_list(self) -> None:
        assert _is_zero([0.0] * 384) is True

    def test_nonzero_list(self) -> None:
        assert _is_zero([0.1] + [0.0] * 383) is False


class TestHybridScore:
    def test_metadata_only(self) -> None:
        meta = np.array([1.0, 0.0, 0.0])
        cand_meta = [1.0, 0.0, 0.0]
        score = _hybrid_score(meta, None, cand_meta, None, 0.7)
        assert score == pytest.approx(1.0)

    def test_audio_only(self) -> None:
        audio = np.array([0.0, 1.0, 0.0])
        cand_audio = [0.0, 1.0, 0.0]
        score = _hybrid_score(None, audio, None, cand_audio, 0.7)
        assert score == pytest.approx(1.0)

    def test_hybrid_weighting(self) -> None:
        meta = np.array([1.0, 0.0])
        audio = np.array([1.0, 0.0])
        cand_meta = [1.0, 0.0]
        cand_audio = [0.0, 1.0]  # orthogonal to audio query
        # meta contributes 0.3 * 1.0 = 0.3, audio contributes 0.7 * 0.0 = 0.0
        score = _hybrid_score(meta, audio, cand_meta, cand_audio, 0.7)
        assert score == pytest.approx(0.3)

    def test_no_vectors(self) -> None:
        score = _hybrid_score(None, None, None, None, 0.7)
        assert score == 0.0
