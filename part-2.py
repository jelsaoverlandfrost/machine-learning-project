import argparse


def parse_training_file(file_name):
    print('Parsing training file...')
    training_file = open(file_name, 'r', encoding='utf-8')
    list_words = []
    list_tags = []
    for line in training_file:
        if line != '' and line != '\n':
            content = line.split()
            word = line[:(len(line) - len(content[-1])) - 2]
            tag = content[-1]
            list_words.append(word)
            list_tags.append(tag)
    print('Parsing finished.')
    return list_words, list_tags


def parse_test_file(filename):
    print('Parsing testing file...')
    testing_file = open(filename, 'r', encoding='utf-8')
    list_words = []
    list_current_sentence = []
    for line in testing_file:
        if line != '\n' and line != '':
            list_current_sentence.append(line.split()[0])
        elif line == '\n':
            list_words.append(list_current_sentence)
        else:
            break
    print('Parsing finished.')
    return list_words


# This function estimates the emission parameter from the training set using MLE
# list_tags = nested list of sequences of y sequences
# list_words = nested list of sequences of x sequences
# Assume valid sequence
def train_emission(list_words, list_tags):
    print('Calculating Emission...')
    dictionary_words = {}
    dictionary_tags = {}
    for i in range(len(list_tags)):
        if list_words[i] not in dictionary_words:
            new_pair = {list_tags[i]: 1.0}
            dictionary_words[list_words[i]] = new_pair
        if list_tags[i] not in dictionary_words[list_words[i]]:
            dictionary_words[list_words[i]][list_tags[i]] = 1.0
        else:
            dictionary_words[list_words[i]][list_tags[i]] = dictionary_words[list_words[i]][list_tags[i]] + 1
        if list_tags[i] not in dictionary_tags:
            dictionary_tags[list_tags[i]] = 1.0
        else:
            dictionary_tags[list_tags[i]] = dictionary_tags[list_tags[i]] + 1.0
    print('Finished.')
    return dictionary_words, dictionary_tags


def test_emission(list_tags, list_words, tag, word, k):
    word_counter = 0.0
    tag_counter = 0.0
    for i in range(len(list_tags)):
        if (list_words[i] == word) and (list_tags[i] == tag):
            word_counter = word_counter + 1
        if i == tag:
            tag_counter = tag_counter + 1
    if word_counter != 0:
        return float(word_counter) / (tag_counter + k)
    else:
        return k / (tag_counter + k)


def argmax(list_tags, list_words, x, k, tags):
    max_arg = 0
    tag = ""
    for i in tags:
        if test_emission(list_tags, list_words, i, x, k) > max_arg:
            max_arg = test_emission(list_tags, list_words, i, x, k)
            tag = i
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
        list_words, list_tags = parse_training_file(train_file_name)
        list_test_seq = parse_test_file(test_file_name)
        dict_words, dict_tags = train_emission(list_words, list_tags)
        predict = {}
        number = 0
        k = count_unknown(list_words, list_test_seq)
        for seq in list_test_seq:
            print('check')
            for word in seq:
                if word not in predict:
                    tag, max_arg = argmax(list_tags, list_words, word, k, dict_tags.keys())
                    predict[word] = tag
                result_file.write(word + " " + predict[word] + "\n")
            number = number + 1
            result_file.write('\n')
    finally:
        result_file.close()


part_2('data/SG/dev.in', 'data/SG/train', 'data/SG/dev.p2.out')
