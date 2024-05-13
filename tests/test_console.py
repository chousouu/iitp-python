"""Console Tests."""

from unittest.mock import Mock

from click.testing import CliRunner
import pytest
from pytest_mock import MockFixture  # type: ignore
import requests  # type: ignore

import fasthough.console as console


@pytest.fixture
def mock_wikipedia_random_page(mocker: MockFixture) -> Mock:
    """Wikipedia random page Mock."""
    return mocker.patch("fasthough.wikipedia.random_page")


@pytest.mark.e2e
def test_main_succeeds_in_production_env(runner: CliRunner) -> None:
    """Test 1. Production status code success."""
    result = runner.invoke(console.main)
    assert result.exit_code == 0


def test_main_succeeds(runner: CliRunner, mock_requests_get: Mock) -> None:
    """Test 2. Main succeed."""
    result = runner.invoke(console.main)
    assert result.exit_code == 0


def test_main_prints_title(runner: CliRunner, mock_requests_get: Mock) -> None:
    """Test 3. Title."""
    result = runner.invoke(console.main)
    assert "Lorem Ipsum" in result.output


def test_main_invokes_requests_get(runner: CliRunner, mock_requests_get: Mock) -> None:
    """Test 4. Page requests."""
    runner.invoke(console.main)
    assert mock_requests_get.called


def test_main_uses_en_wikipedia_org(runner: CliRunner, mock_requests_get: Mock) -> None:
    """Test 5. English Wikipedia."""
    runner.invoke(console.main)
    args, _ = mock_requests_get.call_args
    assert "en.wikipedia.org" in args[0]


def test_main_fails_on_request_error(
    runner: CliRunner, mock_requests_get: Mock
) -> None:
    """Test 6. Request error."""
    mock_requests_get.side_effect = Exception("Failed to read.")
    result = runner.invoke(console.main)
    assert result.exit_code


def test_main_prints_message_on_request_error(
    runner: CliRunner, mock_requests_get: Mock
) -> None:
    """Test 7. Print Error."""
    mock_requests_get.side_effect = requests.RequestException
    result = runner.invoke(console.main)
    assert "Error" in result.output


def test_main_uses_specified_language(
    runner: CliRunner, mock_wikipedia_random_page: Mock
) -> None:
    """Test 8. Other languages."""
    runner.invoke(console.main, ["--language=pl"])
    mock_wikipedia_random_page.assert_called_with(language="pl")
