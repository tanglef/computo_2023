from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import json
    from pathlib import Path


class Dataset(BaseDataset):

    name = "LabelMe"
    requirements = ["pip:json", "numpy"]
    install_cmd = "conda"

    def votes(self):
        current_path = Path(__file__).parent
        with open(
            current_path
            / ".."
            / ".."
            / "datasets"
            / "labelme"
            / "answers.json",
            "r",
        ) as f:
            votes = json.load(f)
        true_labels = np.load(current_path / "truth_labelme.npy")
        self.answers = votes
        self.ground_truth = true_labels

    def get_data(self):
        self.votes()
        return dict(
            votes=self.answers,
            ground_truth=self.ground_truth,
            n_worker=77,
            n_task=1000,
            n_classes=8,
        )
