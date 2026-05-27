import json
import sys
import types

import pytest

from tests.helpers import load_module


def test_feature_planner_respects_dependencies_and_vram_budget():
    planner = load_module("src.utils.feature_planner", "src/utils/feature_planner.py")
    tasks = [
        planner.Task("a", vram_gb=3.0, duration_est_sec=10.0, flops=10.0),
        planner.Task("b", vram_gb=3.0, duration_est_sec=20.0, flops=20.0),
        planner.Task("c", vram_gb=2.0, duration_est_sec=5.0, flops=5.0, depends_on=["a"]),
    ]

    plan = planner.FeaturePlanner(vram_budget_gb=4.0, max_parallel=2).schedule(tasks)

    step_names = [[scheduled.task.name for scheduled in step] for step in plan.steps]
    assert step_names == [["b"], ["a"], ["c"]]
    assert step_names.index(["a"]) < step_names.index(["c"])
    assert plan.total_time_sec == 35.0
    assert plan.peak_vram_gb <= 4.0
    assert plan.total_flops == 35.0


def test_feature_planner_detects_unknown_dependency_and_cycle():
    planner = load_module("src.utils.feature_planner", "src/utils/feature_planner.py")

    with pytest.raises(ValueError):
        planner.FeaturePlanner().schedule([planner.Task("a", depends_on=["missing"])])

    with pytest.raises(RuntimeError):
        planner.FeaturePlanner().schedule([planner.Task("a", depends_on=["b"]), planner.Task("b", depends_on=["a"])])


def test_feature_planner_ljf_speedup_and_summary(capsys):
    planner = load_module("src.utils.feature_planner", "src/utils/feature_planner.py")
    tasks = [planner.Task("short", duration_est_sec=2.0), planner.Task("long", duration_est_sec=5.0)]

    p = planner.FeaturePlanner(vram_budget_gb=8.0, max_parallel=2).schedule_longest_job_first(tasks)
    text = p.print_summary()

    assert p.steps[0][0].task.name == "long"
    assert "Total estimated time" in text
    assert "Execution steps" in capsys.readouterr().out
    assert planner.FeaturePlanner(vram_budget_gb=8.0, max_parallel=2).speedup(tasks) > 1.0


def test_feature_planner_default_tasks_and_completed_filter():
    planner = load_module("src.utils.feature_planner", "src/utils/feature_planner.py")
    tasks = planner.build_default_feature_tasks(video_duration_sec=100)
    filtered = planner.tasks_minus_completed(tasks, ["face_detection", "source_separation"])

    assert len(tasks) > 5
    assert all(task.name != "face_detection" for task in filtered)
    pose = next(task for task in filtered if task.name == "pose_estimation")
    audio = next(task for task in filtered if task.name == "audio_features")
    assert "face_detection" not in pose.depends_on
    assert "source_separation" not in audio.depends_on


def test_feature_planner_detection_helpers(monkeypatch):
    planner = load_module("src.utils.feature_planner", "src/utils/feature_planner.py")

    monkeypatch.setattr(planner.subprocess, "check_output", lambda *a, **k: "4096\n")
    assert planner.detect_vram_budget_gb(fallback=8.0) == 4.0

    monkeypatch.setattr(planner.subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert planner.detect_vram_budget_gb(fallback=7.0) == 7.0

    monkeypatch.setenv("SEENX_FLOPS_PER_SEC", "123.5")
    assert planner.detect_flops_per_sec_guess() == 123.5

    monkeypatch.setenv("SEENX_FLOPS_PER_SEC", "bad")
    assert planner.detect_flops_per_sec_guess(fallback=9.0) == 9.0


def test_config_loads_json_resolves_devices_and_batch_sizes(tmp_path, monkeypatch):
    config_mod = load_module("src.utils.config", "src/utils/config.py")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"device": "gpu", "batch_size": 16, "text_prob_batch_size": None, "face_screen_batch_size": 0.5}), encoding="utf-8")

    autobatch = types.ModuleType("src.utils.autobatch")
    autobatch.resolve_batch_size = lambda value, default, task: int(default / 2)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.utils.autobatch", autobatch)

    cfg = config_mod.Config(str(path))

    assert cfg.get("device") == "cuda"
    assert cfg.get("batch_size") == 16
    assert cfg.get("text_prob_batch_size") == 4
    assert cfg.get("face_screen_batch_size") == 16
    assert cfg.get("missing", "fallback") == "fallback"


def test_config_errors_on_missing_file_and_get_device(monkeypatch, tmp_path):
    config_mod = load_module("src.utils.config", "src/utils/config.py")

    with pytest.raises(FileNotFoundError):
        config_mod.Config(str(tmp_path / "missing.json"))

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert config_mod.get_device({"device": "auto"}) == "cuda"
    assert config_mod.get_device({"device": 123}) == "cpu"


def test_transcript_cache_uses_absolute_path_and_clear_cache(monkeypatch, tmp_path):
    for package_name in ("src", "src.utils"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    config_mod = types.ModuleType("src.utils.config")
    config_mod.Config = dict  # type: ignore[attr-defined]
    logger_mod = types.ModuleType("src.utils.logger")
    logger_mod.Logger = lambda show=True: types.SimpleNamespace(  # type: ignore[attr-defined]
        get_logger=lambda: types.SimpleNamespace(info=lambda *a, **k: None)
    )

    calls = {"n": 0}

    class FakeModel:
        def transcribe(self, video_path, **kwargs):
            calls["n"] += 1
            return {"segments": [{"text": video_path}], "kwargs": kwargs}

    whisper = types.ModuleType("whisper")
    whisper.load_model = lambda *a, **k: FakeModel()  # type: ignore[attr-defined]
    torch = types.ModuleType("torch")
    torch.device = lambda value: value  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "src.utils.config", config_mod)
    monkeypatch.setitem(sys.modules, "src.utils.logger", logger_mod)
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    monkeypatch.setitem(sys.modules, "torch", torch)

    cache = load_module("src.utils.transcript_cache", "src/utils/transcript_cache.py")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")
    cfg = {"device": "cpu", "whisper_model_size": "tiny"}

    first = cache.get_transcript(str(video), cfg)
    second = cache.get_transcript(str(video), cfg)
    cache.clear_cache()
    third = cache.get_transcript(str(video), cfg)

    assert first == second
    assert first["kwargs"]["word_timestamps"] is True
    assert first["kwargs"]["language"] == "ru"
    assert third == first
    assert calls["n"] == 2
