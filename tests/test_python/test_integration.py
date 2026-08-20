import gymnasium as gym
import pytest

import env_stableretro_turbo.testing.tools
from tests.test_python import all_games


@pytest.mark.parametrize("game_name, integration_type", all_games)
def test_data(game_name, integration_type):
    warnings, errors = env_stableretro_turbo.testing.tools.verify_data(
        game_name,
        integration_type,
    )
    for file, warning in warnings:
        gym.logger.warn(f"{file}: {warning}")
    assert len(errors) == 0


@pytest.mark.parametrize("game_name, integration_type", all_games)
def test_scenario(game_name, integration_type):
    warnings, errors = env_stableretro_turbo.testing.tools.verify_scenario(
        game_name,
        integration_type,
    )
    for file, warning in warnings:
        gym.logger.warn(f"{file}: {warning}")
    assert len(errors) == 0


def test_missing():
    missing = env_stableretro_turbo.testing.tools.scan_missing()
    assert len(missing) == 0


@pytest.mark.parametrize("game_name, integration_type", all_games)
def test_default_states(game_name, integration_type):
    warnings, errors = env_stableretro_turbo.testing.tools.verify_default_state(
        game_name,
        integration_type,
    )
    for file, warning in warnings:
        gym.logger.warn(f"{file}: {warning}")
    assert len(errors) == 0
