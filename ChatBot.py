#importing libraries
import tensorflow as tf
import numpy as np
import re
import time

###DATA-PREPROCESSING

#importing the dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

#creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

#creating a list of all of the conversations
conversation_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversation_ids.append(_conversation.split(','))

#getting separately the questions and the answers
questions = []
answers = []
for conversation in conversation_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

#cleaning the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

#cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

#creating a dictionary to map each word's occurances
word_count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
            
for answer in clean_answers:
    for word in answer.split():
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
            
#creating two dictionaries that map the question words and the answer word with a unique integer
threshold = 20
questionwords2int = {}
word_number = 0
for word, count in word_count.items():
    if count >= threshold:
        questionwords2int[word] = word_number
        word_number += 1

answerwords2int = {}
word_number = 0
for word, count in word_count.items():
    if count >= threshold:
        answerwords2int[word] = word_number
        word_number += 1

#adding the last tokens to these two dictionaries
tokens =['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionwords2int[token] = len (questionwords2int) + 1
for token in tokens:
    answerwords2int[token] = len(answerwords2int) + 1

#creating the inverse dictionary of the answerwords2int dictionary
answerints2word = {w_i : w for w, w_i in answerwords2int.items()}

#adding the end of line token at the end of every element in clean_answers
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

#translating all the arguments and the answers into integers and replacing all the words that were filtered out by <OUT>
questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int['<OUT>'])
        else:
            ints.append(questionwords2int[word])
    questions_to_int.append(ints)

answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerwords2int:
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append(answerwords2int[word])
    answers_to_int.append(ints)

#sorted questions and answers by the length of questions
sorted_clean_questions =  []
sorted_clean_answers = []
for length in range(1,25 + 1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])


#creating placeholder for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob

#preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(target, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

































