import json
from collections import defaultdict, Counter 
from pprint import pprint


PAD_INDEX = 0
PAD_WORD = '<PAD>'

START_INDEX = 1
START_WORD = '<STR>'

END_INDEX = 2
END_WORD = '<END>'

UNK_INDEX = 3
UNK_WORD = '<UNK>'


class BaseData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.parse_data()

    def parse_data(self):
        data = defaultdict(dict)
        text_data = []
        with open(self.data_dir, 'r') as jsonfile:
            for reviewID, line in enumerate((jsonfile)):
                line_data = json.loads(line)
                data[reviewID]["userID"] = line_data["reviewerID"]
                data[reviewID]["itemID"] = line_data["asin"]
                data[reviewID]["helpful"] = line_data["helpful"]
                data[reviewID]["reviewText"] = line_data["reviewText"]
                [text_data.append(word) for word in line_data["reviewText"].split()]
                data[reviewID]["rating"] = line_data["overall"]

        self.word_counter = Counter(text_data)
        print(self.word_counter)

        return data

    
        
    
        
            




if __name__ == "__main__":
    data = BaseData("./data/Musical_Instruments_5.json")