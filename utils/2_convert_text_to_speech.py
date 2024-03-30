import json
import os
import sys


class TextToSpeechConverter(L.LightningDataModule):
    def __init__(self, max_length, batch_size, lm_model=None):
        self.liputan6_dir = "datasets/liputan6_data/canonical"

    def join_paragraphs(self, paragraphs):
        # Join List of paragraphs to string paragraphs
        string_paragraph = ""
        for parag in paragraphs:
            for kal in parag:
                kalimat = " ".join(kal)
                string_paragraph += kalimat

        return string_paragraph

    def join_summary(self, summaries):
        string_summary = ""
        # Join Kalimat dalam Satu Paragraf
        for sumr in summaries:
            kal_sum = " ".join(sumr)
            string_summary += kal_sum

        return string_summary

    def load_data(self, flag):
        list_files = os.listdir(f"{self.liputan6_dir}/{flag}")

        datasets = []
        for fl in list_files:
            print(f"{self.liputan6_dir}/{flag}/{fl}")
            with open(
                f"{self.liputan6_dir}/{flag}/{fl}", "r", encoding="utf-8"
            ) as json_reader:
                # load file jsonl (jsonl = kumpulan file json format di gabung jadi satu file)

                data_raw = json_reader.readlines()
                # json_raw = [json.loads(jline) for jline in json_reader.readlines().rstrip().splitlines()]
                # print(data_raw)
                # sys.exit()

            json_raw = []
            for dd in data_raw:

                data_line = json.loads(dd)
                paragraphs = self.join_paragraphs(data_line["clean_article"])
                summary = self.join_paragraphs(data_line["clean_summary"])

                data = {
                    "id": data_line["id"],
                    "paragraphs": paragraphs,
                    "summary": summary,
                }

                json_raw.append(data)

            datasets += json_raw

        # print(len(datasets))

        return datasets

    def preprocessor(self):
        preprocessed_dir = "./datasets/preprocessed/liputan6/"

        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)

        if not os.path.exists("datasets/preprocessed/liputan6/train.pt"):
            # raw data masih dalam format list
            raw_train_data = self.load_data(flag="train")
            print("generated row data")
            print(raw_train_data)
            raw_val_data = self.load_data(flag="dev")
            raw_test_data = self.load_data(flag="test")

            # list => tensor dataset
            train_data = self.list2tensor(data=raw_train_data)
            val_data = self.list2tensor(raw_val_data)
            test_data = self.list2tensor(raw_test_data)

            # save
            torch.save(train_data, preprocessed_dir + "train.pt")
            torch.save(val_data, preprocessed_dir + "val.pt")
            torch.save(test_data, preprocessed_dir + "test.pt")
        else:
            # load existing preprocessed data
            train_data = torch.load(preprocessed_dir + "train.pt")
            val_data = torch.load(preprocessed_dir + "val.pt")
            test_data = torch.load(preprocessed_dir + "test.pt")

        return train_data, val_data, test_data


if __name__ == "__main__":
    pre = TextToSpeechConverter(
        max_length=512,
        batch_size=5,
    )
    pre.setup(stage="fit")
    for data in pre.train_dataloader():
        print(len(data))
        sys.exit()
