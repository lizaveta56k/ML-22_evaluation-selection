
from click.testing import CliRunner
import pytest

from poetry_task_9.train import train

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_test_split_ratio(
    runner: CliRunner
) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--test-split-ratio",
            "42",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--test-split-ratio'" in result.output

def test_error_for_invalid_max_iter(
    runner: CliRunner
) -> None:
    """It fails when max iter is lower than 1."""
    result = runner.invoke(
        train,
        [
            "--max-iter",
            "0",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--max-iter'" in result.output

def test_error_for_invalid_n_init(
    runner: CliRunner
) -> None:
    """It fails when n init is lower than 1."""
    result = runner.invoke(
        train,
        [
            "--n_init",
            "0",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--n_init'" in result.output

def test_error_for_invalid_n_clusters(
    runner: CliRunner
) -> None:
    """It fails when n clusters is lower than 1."""
    result = runner.invoke(
        train,
        [
            "--n_clusters",
            "0",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--n_clusters'" in result.output