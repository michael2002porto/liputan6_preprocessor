import json
import os
import sys


class TextToSpeechConverter:
    def __init__(self):
        self.liputan6_dir = "datasets/liputan6_data/canonical"
        self.preprocessed_dir = "./datasets/liputan6_audio/"

    def join_article(self, article):
        string_article = ""
        # Join Kalimat dalam Satu Paragraf
        for sentence in article:
            kal_sum = " ".join(sentence)
            string_article += kal_sum

        return string_article

    def load_data(self, flag):
        list_files = os.listdir(f"{self.liputan6_dir}/{flag}")

        for fl in list_files:
            print(f"{self.liputan6_dir}/{flag}/{fl}")
            with open(
                f"{self.liputan6_dir}/{flag}/{fl}", "r", encoding="utf-8"
            ) as json_reader:
                # load file jsonl (jsonl = kumpulan file json format di gabung jadi satu file)
                data_raw = json_reader.readlines()

            for dd in data_raw:
                data_line = json.loads(dd)
                paragraphs = self.join_article(data_line["clean_article"])

                print(paragraphs)
                sys.exit()

    def text_to_speech_converter(self):
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)


if __name__ == "__main__":
    pre = TextToSpeechConverter()
    pre.load_data(flag="train")
    for data in pre.train_dataloader():
        print(len(data))
        sys.exit()
