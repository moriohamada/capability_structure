DATA_PATH = "./data/merged_benchmarks.csv"
PLOTS_PATH = "./plots/"
ANALYSIS_OPS = {
    "minModelsPerBench": 30,
    "minBenchmarksPerModel": 8,
    "nNullIter": 500,
    "kNN_k": 5,                     # this doesnt seem to change too much in 3-8 range. using 5 like epoch.
    "sigThresh": 0.01,
    "nTopCreators": 3,              # for creating dummies. just only 3 creators had 5+ models
}