from autocluster_omp.cli import main


if __name__ == "__main__":
    for algorithm in ("kmeans", "fuzzy-cmeans"):
        for preset in ("small", "medium", "large"):
            main(["experiment", "--algorithm", algorithm, "--preset", preset])
