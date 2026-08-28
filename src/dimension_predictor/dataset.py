from torch.utils.data import Dataset


IGNORE_INDEX = -100


class MultiDimensionDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        config,
        labels_config
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.tokenizer = tokenizer

        self.max_length = (
            config["model"]["max_length"]
        )

        self.text_column = (
            config["data"]["text_column"]
        )

        self.source_column = (
            config["data"]["source_column"]
        )

        self.dimensions = (
            labels_config["dimensions"]
        )

        # Statuts internes :
        # absent est traité par présence.
        self.status_vocabulary = [
            "validee",
            "manquante",
            "preuve_manquante",
            "favorable",
            "en_effort",
            "defavorable",
        ]

        self.status2id = {
            status: index
            for index, status
            in enumerate(
                self.status_vocabulary
            )
        }

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        encoding = self.tokenizer(
            str(row[self.text_column]),
            truncation=True,
            max_length=self.max_length,
            padding=False
        )

        presence_labels = []
        status_labels = []

        for dimension in self.dimensions:

            status = str(
                row[dimension["id"]]
            )

            if status == "absent":

                presence_labels.append(0)

                status_labels.append(
                    IGNORE_INDEX
                )

            else:

                presence_labels.append(1)

                status_labels.append(
                    self.status2id[status]
                )

        encoding[
            "presence_labels"
        ] = presence_labels

        encoding[
            "status_labels"
        ] = status_labels

        return encoding
