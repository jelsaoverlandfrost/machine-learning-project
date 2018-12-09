import argparse
from collections import defaultdict


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
        print('Predicting tags...')
        for seq in list_test_seq:
            for word in seq:
                if word not in predict:
                    tag, max_arg = argmax(word, dict_tags.keys(), dict_emission)
                    predict[word] = tag
                result_file.write(word + " " + predict[word] + '\n')
            number = number + 1
            result_file.write('\n')
        print('Finished. Output file written to: ' + result_file_name)
    finally:
        result_file.close()


# Read entities from predcition
def get_predicted(predicted, answers=defaultdict(lambda: defaultdict(defaultdict))):
    example = 0
    word_index = 0
    entity = []
    last_ne = "O"
    last_sent = ""
    last_entity = []

    answers[example] = []
    for line in predicted:
        line = line.strip()
        if line.startswith("##"):
            continue
        elif len(line) == 0:
            if entity:
                answers[example].append(list(entity))
                entity = []

            example += 1
            answers[example] = []
            word_index = 0
            last_ne = "O"
            continue
        else:
            split_line = line.split(separator)
            # word = split_line[0]
            value = split_line[outputColumnIndex]
            ne = value[0]
            sent = value[2:]

            last_entity = []

            # check if it is start of entity
            if ne == 'B' or (ne == 'I' and last_ne == 'O') or (last_ne != 'O' and ne == 'I' and last_sent != sent):
                if entity:
                    last_entity = list(entity)

                entity = [sent]

                entity.append(word_index)

            elif ne == 'I':
                entity.append(word_index)

            elif ne == 'O':
                if last_ne == 'B' or last_ne == 'I':
                    last_entity = list(entity)
                entity = []

            if last_entity:
                answers[example].append(list(last_entity))
                last_entity = []

        last_sent = sent
        last_ne = ne
        word_index += 1

    if entity:
        answers[example].append(list(entity))

    return answers


# Read entities from gold data
def get_observed(observed):
    example = 0
    word_index = 0
    entity = []
    last_ne = "O"
    last_sent = ""
    last_entity = []

    observations = defaultdict(defaultdict)
    observations[example] = []

    for line in observed:
        line = line.strip()
        if line.startswith("##"):
            continue
        elif len(line) == 0:
            if entity:
                observations[example].append(list(entity))
                entity = []

            example += 1
            observations[example] = []
            word_index = 0
            last_ne = "O"
            continue

        else:
            split_line = line.split(separator)
            word = split_line[0]
            value = split_line[outputColumnIndex]
            ne = value[0]
            sent = value[2:]

            last_entity = []

            # check if it is start of entity, suppose there is no weird case in gold data
            if ne == 'B' or (ne == 'I' and last_ne == 'O') or (last_ne != 'O' and ne == 'I' and last_sent != sent):
                if entity:
                    last_entity = entity

                entity = [sent]

                entity.append(word_index)

            elif ne == 'I':
                entity.append(word_index)

            elif ne == 'O':
                if last_ne == 'B' or last_ne == 'I':
                    last_entity = entity
                entity = []

            if last_entity:
                observations[example].append(list(last_entity))
                last_entity = []

        last_ne = ne
        last_sent = sent
        word_index += 1

    if entity:
        observations[example].append(list(entity))

    return observations


# Print Results and deal with division by 0
def printResult(evalTarget, num_correct, prec, rec):
    if abs(prec + rec) < 1e-6:
        f = 0
    else:
        f = 2 * prec * rec / (prec + rec)
    print('#Correct', evalTarget, ':', num_correct)
    print(evalTarget, ' precision: %.4f' % (prec))
    print(evalTarget, ' recall: %.4f' % (rec))
    print(evalTarget, ' F: %.4f' % (f))


# Compare results bewteen gold data and prediction data
def compare_observed_to_predicted(observed, predicted):
    correct_sentiment = 0
    correct_entity = 0

    total_observed = 0.0
    total_predicted = 0.0

    # For each Instance Index example (example = 0,1,2,3.....)
    for example in observed:

        if example in discardInstance:
            continue

        observed_instance = observed[example]
        predicted_instance = predicted[example]

        # Count number of entities in gold data
        total_observed += len(observed_instance)
        # Count number of entities in prediction data
        total_predicted += len(predicted_instance)

        # For each entity in prediction
        for span in predicted_instance:
            span_begin = span[1]
            span_length = len(span) - 1
            span_ne = (span_begin, span_length)
            span_sent = span[0]

            # For each entity in gold data
            for observed_span in observed_instance:
                begin = observed_span[1]
                length = len(observed_span) - 1
                ne = (begin, length)
                sent = observed_span[0]

                # Entity matched
                if span_ne == ne:
                    correct_entity += 1

                    # Entity & Sentiment both are matched
                    if span_sent == sent:
                        correct_sentiment += 1

    print()
    print('#Entity in gold data: %d' % (total_observed))
    print('#Entity in prediction: %d' % (total_predicted))
    print()

    prec = correct_entity / total_predicted
    rec = correct_entity / total_observed
    printResult('Entity', correct_entity, prec, rec)
    print()

    prec = correct_sentiment / total_predicted
    rec = correct_sentiment / total_observed
    printResult('Entity Type', correct_sentiment, prec, rec)


parser = argparse.ArgumentParser(description='Provide the training file, the test file, '
                                             'and the output file you wish to use.')
parser.add_argument('-train', type=str)
parser.add_argument('-test', type=str)
parser.add_argument('-output', type=str)
parser.add_argument('-standard', type=str)

args = parser.parse_args()
part_2(args.test, args.train, args.output)

separator = ' '
outputColumnIndex = 1
discardInstance = []
gold = open(args.standard, "r", encoding='UTF-8')
prediction = open(args.output, "r", encoding='UTF-8')
observed = get_observed(gold)
predicted = get_predicted(prediction)
compare_observed_to_predicted(observed, predicted)
