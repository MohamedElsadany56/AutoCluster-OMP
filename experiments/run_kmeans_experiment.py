from autocluster_omp.cli import main


if __name__ == "__main__":
    main(["experiment", "--algorithm", "kmeans", "--preset", "medium"])
