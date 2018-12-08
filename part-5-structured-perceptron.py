# Structured Perceptron for POS Tagging
import numpy as np
import random
from collections import Counter, defaultdict


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
            print("Iteration: " + str(iteration))
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
                print('Accuracy for this sent:' + str(training_accuracy))
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
            '-' in word + '_DASH' + tag
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
            # initialize probability for tags j at position 1 (first word)
            cur_tag = tags[j]
            features = self.get_features(cur_word, cur_tag, self.START)
            feature_weights = 0
            for feature in features:
                feature_weights += self.feature_weights[feature]
            param_matrix[j, 0] = feature_weights

        # iteration step
        # filling the lattice, for every position and every tag find viterbi score Q
        for i in range(1, sentence_length):
            # for every tag
            for j in range(tag_number):
                # checks if we are at end or start
                tag = tags[j]
                best_score = float('-Inf')
                # for every possible previous tag
                for k in range(tag_number):
                    # k=previous tag
                    previous_tag = tags[k]
                    best_before = param_matrix[k, i - 1]  # score until best step before
                    features = self.get_features(words[i], tag, previous_tag)
                    feature_weights = sum((self.feature_weights[x] for x in features))
                    score = best_before + feature_weights
                    if score > best_score:
                        param_matrix[j, i] = score
                        best_score = score
                        back_pointer[j, i] = k  # best tag
        best_id = param_matrix[:, -1].argmax()
        pred_tags = [tags[best_id]]

        for i in range(sentence_length - 1, 0, -1):
            idx = int(back_pointer[best_id, i])
            pred_tags.append(tags[idx])
            best_id = idx

        # return reversed predtags
        # return (words,predtags[::-1])
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


training_sentences = parse_training_file('data/EN/train')
sp = StructuredPerceptron()
sp.train(training_sentences)
sp.predict('test/EN/test.in', 'test/EN/test.out')