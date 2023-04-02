import os
import utils.results as results
import global_settings
import tarfile
import json

TARGET_PATH = global_settings.PATH_RESULTS


def extract(file_name):
    tar = tarfile.open(file_name, "r:gz")
    identifier = next(iter(tar)).name
    tar.extractall(os.path.abspath("/tmp"))

def cleanup(file_name):
    tar = tarfile.open(file_name, "r:gz")
    identifier = next(iter(tar)).name
    tmp_path = os.path.join("/tmp", identifier)
    os.system(f"rm -rf {tmp_path}")

def load_result(file_name):
    extract(file_name)
    tar = tarfile.open(file_name, "r:gz")
    identifier = identifier = next(iter(tar)).name
    tmp_path = os.path.join("/tmp", identifier)
    with open(os.path.join(tmp_path, "result.json")) as f:
        result_json = json.load(f)
    if result_json["type"] == "ResultSample":
        result = results.ResultSample.load_from_file(file_name)
    cleanup(file_name)
    return result

def main():
    fn_results = [os.path.join(TARGET_PATH, file_name) for file_name in os.listdir(TARGET_PATH) if os.path.isfile(os.path.join(TARGET_PATH, file_name)) and file_name.split('.')[-1] == "gz"]
    print(fn_results)
    csv_str = ("{} " * 16 + "{}\n").format(
        "identifier",
        "date",
        "result_type",
        "experiment_type",
        "dataset",
        "dataset_shape",
        "hidden_layers",
        "hidden_neurons",
        "activation",
        "activation_last_layer",
        "num_warmup",
        "statistic",
        "statistic_p",
        "samples_per_chain",
        "identifiable_modes",
        "seed",
        "samples_parameters_shape"
    )
    
    for fn_result in fn_results:
        print(fn_result)
        result = load_result(fn_result)
        entry = ("{} " * 16 + "{}\n").format(
            result.identifier,
            result.date,
            result.__class__.__name__,
            result.experiment_type,
            result.settings.dataset,
            result.dataset.shape,
            result.settings.hidden_layers,
            result.settings.hidden_neurons,
            result.settings.activation,
            result.settings.activation_last_layer,
            result.settings.num_warmup,
            result.settings.statistic,
            result.settings.statistic_p,
            result.settings.samples_per_chain,
            result.settings.identifiable_modes,
            result.settings.seed,
            result.samples["parameters"].shape
        )
        csv_str += entry

    with open(os.path.join(TARGET_PATH, "summary.csv"), "w") as f:
        f.write(csv_str)


if __name__ == "__main__":
    main()
