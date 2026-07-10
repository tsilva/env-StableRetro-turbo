import importlib.util
import sys
from pathlib import Path


def _load_module():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    bench_spec = importlib.util.spec_from_file_location(
        "benchmark_vec_env",
        scripts / "benchmark_vec_env.py",
    )
    bench_module = importlib.util.module_from_spec(bench_spec)
    sys.modules[bench_spec.name] = bench_module
    bench_spec.loader.exec_module(bench_module)

    spec = importlib.util.spec_from_file_location(
        "benchmark_atari_alepy",
        scripts / "benchmark_atari_alepy.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ale_rom_id_from_retro_game():
    bench = _load_module()

    assert bench._ale_rom_id_from_retro_game("Breakout-Atari2600-v0") == "breakout"
    assert bench._ale_rom_id_from_retro_game("MsPacMan-Atari2600-v0") == "ms_pac_man"


def test_summary_reports_sample_stdev():
    bench = _load_module()

    summary = bench._summary([1.0, 2.0, 3.0])
    assert summary["mean"] == 2.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
    assert round(summary["stdev"], 6) == 1.0


def test_dry_run_uses_atari_profile_defaults(capsys):
    bench = _load_module()

    assert bench.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "profile=atari-breakout-diagnostic" in output
    assert "stable_state=Start" in output
    assert "ale_game=breakout" in output
    assert "envs=32 threads=16" in output
