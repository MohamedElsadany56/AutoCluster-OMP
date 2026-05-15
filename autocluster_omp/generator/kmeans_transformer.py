from autocluster_omp.generator.openmp_generator import generate_from_file, generate_openmp_source


def transform_source(source: str, schedule: str = "static") -> tuple[str, list]:
    return generate_openmp_source(source, "kmeans", schedule)


def transform_file(input_path: str, schedule: str = "static") -> tuple[str, list]:
    return generate_from_file(input_path, "kmeans", schedule)
