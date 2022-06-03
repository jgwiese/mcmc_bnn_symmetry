from dataclasses import dataclass
from utils import settings
from typing import List, Any, Dict
import os
import tarfile
import gzip
import shutil
import json
import jax.numpy as jnp


@dataclass
class ResultSample:
    identifier: str
    date: str
    experiment_type: str
    settings: settings.SettingsExperimentSample
    dataset: Any
    samples: Dict

    def _samples_file_names(self):
        samples_file_names = {}
        for key in self.samples.keys():
            samples_file_names[key] = f"samples_{key}.npy"
        return samples_file_names

    def _json(self):
        return {
            "identifier": self.identifier,
            "date": self.date,
            "experiment_type": self.experiment_type,
            "settings": self.settings.__dict__,
            "dataset": "dataset.npy",
            "samples": self._samples_file_names(),
            "type": self.__class__.__name__,
        }
    
    def save(self):
        # temporary folder
        tmp_path = os.path.join("/tmp", self.identifier)
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        
        # write files temporarily to tmp_path
        with open(os.path.join(tmp_path, "result.json"), "w") as f:
            json.dump(self._json(), f)

        jnp.save(os.path.join(tmp_path, "dataset.npy"), self.dataset.data)
        for key in self._samples_file_names().keys():
            jnp.save(os.path.join(tmp_path, self._samples_file_names()[key]), self.samples[key])

        # tar.gz and move to final destination
        file_name = os.path.join(self.settings.output_path, f"{self.identifier}.tar.gz")
        with tarfile.open(file_name, "w:gz") as targz:
                targz.add(tmp_path, arcname=os.path.basename(tmp_path))
        
        # cleanup the tmp_path files
        os.system(f"rm -rf {tmp_path}")
        return file_name

    @staticmethod
    def load_from_file(file_name):
        tar = tarfile.open(file_name, "r:gz")
        identifier = next(iter(tar)).name
        tar.extractall(os.path.abspath("/tmp"))
        
        # load components
        tmp_path = os.path.join("/tmp", identifier)
        with open(os.path.join(tmp_path, "result.json")) as f:
            result_json = json.load(f)

        dataset = jnp.load(os.path.join(tmp_path, result_json["dataset"]))
        samples = {}
        for key in result_json["samples"]:
            samples[key] = jnp.load(os.path.join(tmp_path, result_json["samples"][key]))
        
        # cleanup the tmp_path files
        os.system(f"rm -rf {tmp_path}")

        experiment = settings.SettingsExperimentSample(**result_json["settings"])
        #experiment._samples = samples
        
        return ResultSample(
            identifier=result_json["identifier"],
            date=result_json["date"],
            experiment_type=result_json["experiment_type"],
            settings=experiment,
            dataset=dataset,
            samples=samples
        )
