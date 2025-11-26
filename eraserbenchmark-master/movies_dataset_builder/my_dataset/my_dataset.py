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
        # data_dir is passed automatically by HuggingFace datasets
        # It contains the directory where the dataset script is located
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": dl_manager.download_and_extract("train.csv")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": dl_manager.download_and_extract("val.csv")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": dl_manager.download_and_extract("test.csv")},
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