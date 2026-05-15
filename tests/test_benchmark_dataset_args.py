from pathlib import Path

import pytest

from autocluster_omp.benchmark import runner
from autocluster_omp.cli import build_parser


def test_benchmark_parser_accepts_dataset_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "--algorithm",
            "kmeans",
            "--sequential",
            "seq.c",
            "--manual",
            "manual.c",
            "--auto",
            "auto.c",
            "--dataset",
            "data.csv",
            "--labels",
            "labels.csv",
            "--clusters",
            "15",
            "--features",
            "20",
            "--iterations",
            "30",
            "--points",
            "10000",
            "--repeat",
            "10",
            "--fuzziness",
            "2.5",
        ]
    )

    assert args.dataset == "data.csv"
    assert args.labels == "labels.csv"
    assert args.clusters == 15
    assert args.features == 20
    assert args.iterations == 30
    assert args.points == 10000
    assert args.repeat == 10
    assert args.fuzziness == 2.5


def test_build_runtime_args_for_kmeans_dataset():
    args = runner.build_runtime_args(
        algorithm="kmeans",
        dataset="data.csv",
        clusters=15,
        features=20,
        iterations=30,
        repeat=10,
    )

    assert args == [
        "--dataset",
        "data.csv",
        "--clusters",
        "15",
        "--features",
        "20",
        "--iterations",
        "30",
        "--repeat",
        "10",
    ]


def test_build_runtime_args_for_fcm_dataset_includes_fuzziness():
    args = runner.build_runtime_args(
        algorithm="fuzzy-cmeans",
        dataset="data.csv",
        clusters=15,
        features=20,
        iterations=30,
        repeat=10,
        fuzziness=2.0,
    )

    assert args[-2:] == ["--fuzziness", "2.0"]


def test_dataset_path_validation_reports_missing_file():
    with pytest.raises(FileNotFoundError):
        runner.validate_dataset_inputs("missing.csv")


def test_benchmark_sources_keeps_synthetic_runtime_args_empty(monkeypatch, tmp_path):
    compile_defines = []
    runtime_args_seen = []

    def fake_compile(source, output, openmp=False, defines=None):
        compile_defines.append(dict(defines or {}))
        return Path(output)

    def fake_run(binary, threads, runtime_args=None):
        runtime_args_seen.append(list(runtime_args or []))
        return {"RuntimeSeconds": 1.0, "Checksum": 42.0}

    monkeypatch.setattr(runner, "compile_c_source", fake_compile)
    monkeypatch.setattr(runner, "run_binary", fake_run)

    results = runner.benchmark_sources(
        algorithm="kmeans",
        sequential="seq.c",
        manual="manual.c",
        auto="auto.c",
        threads=[1],
        build_dir=tmp_path,
        points=100,
        clusters=3,
        features=2,
        iterations=4,
    )

    assert all(args == [] for args in runtime_args_seen)
    assert compile_defines[0]["N_POINTS"] == 100
    assert compile_defines[0]["N_CLUSTERS"] == 3
    assert compile_defines[0]["N_FEATURES"] == 2
    assert compile_defines[0]["MAX_ITER"] == 4
    assert results[0].dataset is None


def test_benchmark_sources_passes_dataset_runtime_args(monkeypatch, tmp_path):
    data_path = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"
    data_path.write_text("x,y\n1.0,2.0\n", encoding="utf-8")
    labels_path.write_text("0\n", encoding="utf-8")
    compile_defines = []
    runtime_args_seen = []

    def fake_compile(source, output, openmp=False, defines=None):
        compile_defines.append(dict(defines or {}))
        return Path(output)

    def fake_run(binary, threads, runtime_args=None):
        runtime_args_seen.append(list(runtime_args or []))
        return {"RuntimeSeconds": 1.0, "Checksum": 42.0}

    monkeypatch.setattr(runner, "compile_c_source", fake_compile)
    monkeypatch.setattr(runner, "run_binary", fake_run)

    results = runner.benchmark_sources(
        algorithm="kmeans",
        sequential="seq.c",
        manual="manual.c",
        auto="auto.c",
        threads=[1],
        build_dir=tmp_path,
        dataset=data_path,
        labels=labels_path,
        clusters=2,
        features=2,
        iterations=3,
        repeat=5,
    )

    assert runtime_args_seen[0] == [
        "--dataset",
        str(data_path),
        "--clusters",
        "2",
        "--features",
        "2",
        "--iterations",
        "3",
        "--repeat",
        "5",
    ]
    assert compile_defines[0]["ENABLE_CSV_LOADING"] == 1
    assert results[0].dataset == str(data_path)
    assert results[0].labels == str(labels_path)
