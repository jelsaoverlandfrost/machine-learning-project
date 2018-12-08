import argparse

parser = argparse.ArgumentParser(description='Provide the training file, the test file, '
                                             'and the output file you wish to use.')
parser.add_argument('train', type=str)
parser.add_argument('test', type=str)
parser.add_argument('output', type=str)

args = parser.parse_args()


def parse_training_file(file_name):
    print('Parsing training file...')
    training_file = open(file_name, 'r', encoding='utf-8')
    list_words = []
    list_occurrence_words = []
    list_tags = []
    for line in training_file:
        if line != '' and line != '\n':
            content = line.split()
            word = line[:(len(line) - len(content[-1])) - 2]
            tag = content[-1]
            if word not in list_occurrence_words:
                list_occurrence_words.append(word)
            list_words.append(word)
            list_tags.append(tag)
    print('Parsing finished.')
    return list_words, list_tags, list_occurrence_words


def parse_test_file(filename):
    print('Parsing testing file...')
    testing_file = open(filename, 'r', encoding='utf-8')
    list_words = []
    list_current_sentence = []
    for line in testing_file:
        if line != '\n' and line != '':
            list_current_sentence.append(line.replace('\n', ''))
        elif line == '\n':
            list_words.append(list_current_sentence)
            list_current_sentence = []
        else:
            break
    print('Parsing finished.')
    return list_words


# This function estimates the emission parameter from the training set using MLE
# list_tags = nested list of sequences of y sequences
# list_words = nested list of sequences of x sequences
# Assume valid sequence
# Here, we handle the case where UNK tag appears in the code, generating the emission dictionary.
def train_emission(list_words, list_tags, k=1):
    print('Calculating Emission...')
    dictionary_words = {}
    dictionary_tags = {}
    for i in range(len(list_tags)):
        if list_words[i] not in dictionary_words.keys():
            dictionary_words[list_words[i]] = {list_tags[i]: 1.0}
        if list_tags[i] not in dictionary_words[list_words[i]].keys():
            dictionary_words[list_words[i]][list_tags[i]] = 1.0
        else:
            dictionary_words[list_words[i]][list_tags[i]] = dictionary_words[list_words[i]][list_tags[i]] + 1
        if list_tags[i] not in dictionary_tags.keys():
            dictionary_tags[list_tags[i]] = 1.0
        else:
            dictionary_tags[list_tags[i]] = dictionary_tags[list_tags[i]] + 1.0
    emission = {}
    for word in dictionary_words.keys():
        for tag in dictionary_words[word].keys():
            if word not in emission.keys():
                emission[word] = {}
            emission[word][tag] = float(dictionary_words[word][tag]) / (dictionary_tags[tag] + k)
    emission['#UNK#'] = {}
    for tag in dictionary_tags:
        emission['#UNK#'][tag] = k / (dictionary_tags[tag] + k)
    print('Finished.')
    return dictionary_words, dictionary_tags, emission


def argmax(word, list_tags, emission):
    max_arg = 0
    tag = ""
    if word not in emission.keys():
        word = '#UNK#'
    for iter_tag in list_tags:
        if iter_tag in emission[word].keys():
            if emission[word][iter_tag] > max_arg:
                max_arg = emission[word][iter_tag]
                tag = iter_tag
        else:
            continue
    return tag, max_arg


def count_unknown(list_words, list_test_seq):
    count = 0
    for seq in list_test_seq:
        for word in seq:
            if word not in list_words:
                count = count + 1
    return count


def part_2(test_file_name, train_file_name, result_file_name):
    try:
        result_file = open(result_file_name, "w+", encoding="utf-8")
        list_words, list_tags, list_occurrence = parse_training_file(train_file_name)
        list_test_seq = parse_test_file(test_file_name)
        k = 1
        dict_words, dict_tags, dict_emission = train_emission(list_words, list_tags, k)
        predict = {}
        number = 0
        for seq in list_test_seq:
            for word in seq:
                if word not in predict:
                    tag, max_arg = argmax(word, dict_tags.keys(), dict_emission)
                    predict[word] = tag
                result_file.write(word + " " + predict[word] + '\n')
            number = number + 1
            result_file.write('\n')
    finally:
        result_file.close()


part_2(args[0], args[1], args[2])
