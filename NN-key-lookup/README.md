===============================
This example shows how to use train and use a neural network to learn different questions and their associated responses.
===============================
It is divided into three parts:
1. intents.JSON
2. trainer.py
3. conversation.py

A basic JSON file for training should look like:

{"intents": [
        {"tag": "greeting",
         "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day"],  *>>>>>> These are the example sentences that the network will learn to recognize*
         "responses": ["Shove off.", "I'm doing well."],  *>>>>>> These are the possible responses*
         "context_set": "",
         "context_filter": ""
        }
        ]
 }

 
 The trainer parses all the input sentences into their constituent words, stems them, then creates a Bag-Of-Words from that, which help it to analyze and predict the output based on the input.
