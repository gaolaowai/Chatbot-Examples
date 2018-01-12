# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')
stemmer = LancasterStemmer()
###################################################
# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
##################################################
# restore all of our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
###################################################
# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)
###################################################
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
###################################################
# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
###################################################
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))
###################################################
# load our saved model
model.load('./model.tflearn')
###################################################
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return str(random.choice(i['responses']))

            results.pop(0)
###################################################
#Starting up the server section...
###################################################

import socket
import sys
from _thread import *



host = ''
port = 5555
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.bind((host, port))
except socket.error as e:
    print(str(e))

s.listen(5)
print('Waiting for a connection.')
def threaded_client(conn, addr):
    conn.send(str.encode("Welcome to MaxBot... press \"Enter\" to begin...\nAlso, don\'t press Ctrl+C. You won\'t like it :-( .\n"))

    while True:
        try:
            user_input = conn.recv(2048)
            #some basic validation to get rid of ctrl characters
            if ("0xff".encode('utf-8')) in user_input:
                print("detected!")
            #.decode('utf-8')
            user_input = user_input.decode('utf-8')
            if ("uit" in user_input) or ("eave" in user_input):
                conn.sendall(str.encode("Do you want to leave? (Yes / no)\n"))
                quitreply = conn.recv(1024).decode('utf-8')
                if "es" in quitreply:
                    conn.sendall(str.encode("Alrighty then, out you go...\n"))
                    conn.close()
                else:
                    conn.sendall(str.encode("My bad then, I misunderstood. Let's go ahead and continue then.\n"))
            server_output = str(response(str(user_input), userID=str(addr)))
            server_output = "{}{}".format(server_output, "\n")
            print(user_input, '\n', server_output, '\n')
            if not data:
                break
            conn.sendall(str.encode(server_output))
        except:
            conn.sendall(str.encode("Oh dear... something went wrong on our side.\nApologies, but we're going to need to close this connection.\nPlease try again later.\nThis error has been logged.\nGoodbye!\n"))
            #print("Serverside Error: {}".format(e))
            conn.close()
    conn.close()


while True:

    conn, addr = s.accept()
    print('connected to: '+addr[0]+':'+str(addr[1]))

    start_new_thread(threaded_client,(conn,addr))
