import json
import os
import sys

def join_paragraphs(paragraphs):
    # Join List of paragraphs to string paragraphs
    string_paragraph = ""
    for parag in paragraphs:
        for kal in parag:
            kalimat = " ".join(kal)
            string_paragraph += kalimat 
    
    return string_paragraph

def join_summary(summaries):
    string_summary = ""
    # Join Kalimat dalam Satu Paragraf
    for sumr in summaries:
        kal_sum = " ".join(sumr)
        string_summary += kal_sum

    return string_summary
        
def load_data(flag):
    liputan6_dir = "datasets/liputan6_data/canonical/test"
    list_files = os.listdir(liputan6_dir)
    print(list_files)
    exit()
    datasets = []
    for fl in list_files:
        if flag in fl:
            with open(f"{liputan6_dir}/{fl}", "r", encoding = "utf-8") as json_reader:
                # load file jsonl (jsonl = kumpulan file json format di gabung jadi satu file)
                
                data_raw = json_reader.readlines()                   
                # json_raw = [json.loads(jline) for jline in json_reader.readlines().rstrip().splitlines()]
            
            json_raw = []  
            for dd in data_raw:
                
                data_line = json.loads(dd)
                paragraphs = join_paragraphs(data_line["clean_article"])

                print(paragraphs)
                exit()
                
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
        raw_train_data = load_data(flag = "train")
        raw_val_data = load_data(flag = "dev")
        raw_test_data = load_data(flag = "test")

        # list => tensor dataset
        train_data = list2tensor(data = raw_train_data)
        val_data = list2tensor(raw_val_data)
        test_data = list2tensor(raw_test_data)

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

load_data(flag="train")