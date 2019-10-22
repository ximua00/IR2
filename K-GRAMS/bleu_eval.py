# Importing Libraries
import nltk
from nltk.translate.bleu_score import SmoothingFunction
import os

# Reading the generated folder
generated_sentence_list = []
generated_folder_path = os.path.join('log', 'generated')
for file_name in os.listdir(generated_folder_path):
    with open(os.path.join(generated_folder_path, file_name), 'r') as f:
        sentence = f.read()
        sentence = sentence.split()
        sentence = sentence[1 : -1]
        # print(sentence)
        generated_sentence_list.append(sentence)

# Reading the golden folder
golden_sentence_list = []
golden_folder_path = os.path.join('log', 'golden')
for file_name in os.listdir(golden_folder_path):
    with open(os.path.join(golden_folder_path, file_name), 'r') as f:
        sentence = f.read()
        sentence = sentence.split()
        sentence = sentence[1 : -1]
        # print(sentence)
        golden_sentence_list.append(sentence)

# Computing the BLEU metric
assert len(generated_sentence_list) == len(golden_sentence_list)
bleu_accumulator = 0
for i in range(len(generated_sentence_list)):
    # print("Golden Length : {}, Generated Length : {}".format(len(golden_sentence_list[i]), len(generated_sentence_list)))
    bleu_score = nltk.translate.bleu_score.sentence_bleu([golden_sentence_list[i]], generated_sentence_list[i], smoothing_function = SmoothingFunction().method7)
    bleu_accumulator += bleu_score
print("Average BLEU Score : {}".format(bleu_accumulator / len(generated_sentence_list)))