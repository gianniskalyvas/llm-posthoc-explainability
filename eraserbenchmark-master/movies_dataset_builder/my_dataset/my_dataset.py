import os
import datasets

class MyDataset(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "text": datasets.Value("string"),
                "label": datasets.ClassLabel(names=["negative", "positive"]),
            }),
            supervised_keys=("text", "label"),
        )


    def _split_generators(self, dl_manager):

        data_dir = os.path.dirname(os.path.abspath(__file__))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": f"{data_dir}/train.csv"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": f"{data_dir}/test.csv"},
            ),
        ]

    def _generate_examples(self, filepath):
        import csv

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                yield idx, {
                    "text": row["text"],
                    "label": row["label"],
                }