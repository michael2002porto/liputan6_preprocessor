import json
import os
import sys
import re


class TextToSpeechConverter:
    def __init__(self):
        self.liputan6_dir = "datasets/liputan6_data/canonical"
        self.preprocessed_dir = "./datasets/liputan6_audio/"

        # Dictionary mapping month numbers to Indonesian names
        self.month_names = {
            "1": "januari",
            "2": "februari",
            "3": "maret",
            "4": "april",
            "5": "mei",
            "6": "juni",
            "7": "juli",
            "8": "agustus",
            "9": "september",
            "10": "oktober",
            "11": "november",
            "12": "desember",
        }

    # Function to replace date patterns with month names
    def replace_date(self, match):
        day, month = match.groups()
        return f"{int(day)} {self.month_names[month]}"

    def join_article(self, article):
        string_article = ""
        # Join Kalimat dalam Satu Paragraf
        for sentence in article:
            kal_sum = ""
            i = 0
            for word in sentence:
                kal_sum += word
                next_word = ""
                if i < len(sentence) - 1:
                    next_word = sentence[i + 1]
                if word not in ["(", "{", "["]:
                    if next_word not in [".", ",", ":", ";", ")", "}", "]"]:
                        kal_sum += " "
                i += 1
            string_article += kal_sum

        string_article = string_article.replace("Liputan6. com", "Liputan6.com")

        date_pattern = r"\((\d{1,2})/(\d{1,2})\)"
        string_article = re.sub(date_pattern, self.replace_date, string_article)

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
