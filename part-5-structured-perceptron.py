# Structured Perceptron for POS Tagging
import numpy as np
import random
from collections import Counter, defaultdict
import argparse
import re


def parse_training_file(file_name):
    print('Parsing training file...')
    training_file = open(file_name, 'r', encoding='utf-8')
    words_in_curr_sentence = []
    tags_in_curr_sentence = []
    matrix_mapping = []
    for line in training_file:
        if line != '' and line != '\n':
            content = line.split()
            word = line[:(len(line) - len(content[-1])) - 2]
            tag = content[-1]
            words_in_curr_sentence.append(word)
            tags_in_curr_sentence.append(tag)
        elif line == '\n':
            matrix_mapping.append((words_in_curr_sentence, tags_in_curr_sentence))
            words_in_curr_sentence = []
            tags_in_curr_sentence = []
    print('Parsing finished.')
    return matrix_mapping


class StructuredPerceptron(object):

    def __init__(self, seed=1):
        self.feature_weights = defaultdict(float)
        self.tags = []
        self.START = 'START'
        self.END = 'END'
        self.random_seed = seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def train(self, train_mapping, iterations=5, learning_rate=0.2):
        average_weights = Counter()
        for iteration in range(iterations):
            print("Training, this is iteration: " + str(iteration))
            for sentence_tuple in train_mapping:
                correct = 0
                list_words, list_tags = sentence_tuple
                for tag in list_tags:
                    if tag not in self.tags:
                        self.tags.append(tag)
                prediction = self.predict_tags(list_words)
                global_gold_features = self.get_global_features(list_words, list_tags)
                global_predict_features = self.get_global_features(list_words, prediction)
                for feature, count in global_gold_features.items():
                    self.feature_weights[feature] += (learning_rate * count)
                for feature, count in global_predict_features.items():
                    self.feature_weights[feature] -= (learning_rate * count)
                for i in range(len(prediction)):
                    if prediction[i] == list_tags[i]:
                        correct += 1
                training_accuracy = correct / len(list_tags)
            average_weights.update(self.feature_weights)
            random.shuffle(train_mapping)
        self.feature_weights = average_weights

    def get_global_features(self, words, tags):
        feature_counts = Counter()
        for i in range(len(words)):
            if i == 0:
                previous_tag = self.START
            else:
                previous_tag = tags[i - 1]
            feature_counts.update(self.get_features(words[i], tags[i], previous_tag))
        return feature_counts

    def get_features(self, word, tag, previous_tag):
        features = [
            tag,
            previous_tag + '+' + tag,
            word + '+' + tag,
            'www' in word + '_WWW' + tag,
            '-' in word + '_DASH' + tag,
            str(bool(re.search(r'\d', word))) + '_NUMBER' + tag
        ]
        return features

    def predict_tags(self, words):
        sentence_length = len(words)
        tag_number = len(self.tags)
        tags = list(self.tags)

        param_matrix = np.ones((len(self.tags), sentence_length)) * float('-Inf')
        back_pointer = np.ones((len(self.tags), sentence_length), dtype=np.int16) * -1

        cur_word = words[0]
        for j in range(tag_number):
            cur_tag = tags[j]
            features = self.get_features(cur_word, cur_tag, self.START)
            feature_weights = 0
            for feature in features:
                feature_weights += self.feature_weights[feature]
            param_matrix[j, 0] = feature_weights

        for i in range(1, sentence_length):
            for j in range(tag_number):
                tag = tags[j]
                best_score = -1
                for k in range(tag_number):
                    previous_tag = tags[k]
                    best_before = param_matrix[k, i - 1]  # score until best step before
                    features = self.get_features(words[i], tag, previous_tag)
                    feature_weights = sum((self.feature_weights[x] for x in features))
                    score = best_before + feature_weights
                    if score > best_score:
                        param_matrix[j, i] = score
                        best_score = score
                        back_pointer[j, i] = k
        best_id = param_matrix[:, -1].argmax()
        pred_tags = [tags[best_id]]

        for i in range(sentence_length - 1, 0, -1):
            idx = int(back_pointer[best_id, i])
            pred_tags.append(tags[idx])
            best_id = idx
        return pred_tags[::-1]

    def predict(self, file_name, out_file_name):
        test_file = open(file_name, 'r', encoding='utf-8')
        output_file = open(out_file_name, 'w+', encoding='utf-8')
        test_sentence = []
        test_data = []
        for line in test_file:
            if line != '\n' and line != '':
                test_sentence.append(line.replace('\n', ''))
            elif line == '\n':
                test_data.append(test_sentence)
                test_sentence = []
        for sentence in test_data:
            predicted_tags = self.predict_tags(sentence)
            for i in range(len(predicted_tags)):
                output_file.write(sentence[i] + ' ' + predicted_tags[i] + '\n')
            output_file.write('\n')


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
parser.add_argument('-iterations', type=int, default=20)
parser.add_argument('-rate', type=float, default=0.2)

args = parser.parse_args()

training_sentences = parse_training_file(args.train)
sp = StructuredPerceptron()
sp.train(training_sentences, iterations=args.iterations, learning_rate=args.rate)
sp.predict(args.test, args.output)

separator = ' '
outputColumnIndex = 1
discardInstance = []
gold = open(args.standard, "r", encoding='UTF-8')
prediction = open(args.output, "r", encoding='UTF-8')
observed = get_observed(gold)
predicted = get_predicted(prediction)
compare_observed_to_predicted(observed, predicted)