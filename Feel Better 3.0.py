import openai
import tkinter
import webbrowser
import csv
import re
import nltk
import urllib.request
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.naive_bayes import MultinomialNB

# type your openai api key below
openai.api_key = ("")
# Reading the database from the internet
url = "https://raw.githubusercontent.com/altarknmmn/Feel-Better/main/sentence.csv"
res = urllib.request.urlopen(url)
lines = [l.decode('utf-8') for l in res.readlines()]


nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()

top = tkinter.Tk()
top.state('zoomed')
top.title("Feel Better")


def open_browser(x):
    if x == 1:
        webbrowser.open_new("https://discord.com/channels/940676870758871091/940676870758871094")
    elif x == 2:
        webbrowser.open_new("https://discord.com/channels/940677796487888966/940677797200937002")
    elif x == 3:
        webbrowser.open_new("https://discord.com/channels/940677881489670154/940677882617954336")
    elif x == 4:
        webbrowser.open_new("https://discord.com/channels/940677985353211925/940677985877524532")
    elif x == 5:
        webbrowser.open_new("https://discord.com/channels/940678232146079834/940678232146079837")


# Function that tokenizes the words in the database


def process_text(text):
    text = re.sub('[^A-Za-z]', ' ', text.lower())

    tokenized_text = word_tokenize(text)

    clean_text = [
        stemmer.stem(word) for word in tokenized_text
        if word not in stopwords.words('english')
    ]

    return clean_text


starting_text = tkinter.Label(top, text="Feel Better", font=("Helvetica", 48))
starting_text.pack()


def start():
    # Building the GUI
    starting_text.pack_forget()
    button_for_start.pack_forget()
    label = tkinter.Label(top, text="You can type here what troubles you:", font=("Helvetica", 24))
    label.pack(side="top")
    entry = tkinter.Entry(top, width=50, borderwidth=5, bd=5, font=("Helvetica", 24))
    entry.pack(side="top")
    starter_button = tkinter.Button(top, text="Enter", font=("Helvetica", 24), command=lambda: feel_better_engine())
    starter_button.pack(side="top")
    tkinter.Button(top, text="Quit", font=("Helvetica", 24), command=top.destroy).pack(side="bottom")

    def feel_better_engine():
        label.pack_forget()
        entry.pack_forget()
        starter_button.pack_forget()
        # Taking the user's input
        examined_text = entry.get()
        question = str(examined_text)
        # Creating a human-like response using OpenAI's library
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}],
                                                temperature=0.8, max_tokens=1024)
        feel_better_answer = response.choices[0].message.content

        sentence = [row for row in csv.reader(lines)]
        sentence = sentence[1:]
        # 01/2022

        texts = [row[0] for row in sentence]
        topics = [row[1] for row in sentence]

        texts = [" ".join(process_text(text)) for text in texts]
        examined_texts = sent_tokenize(examined_text)
        my_list = []

        for sent in texts:
            words = sent.split()
            for one_word in words:
                my_list.append(one_word)

        my_list = list(set(my_list))
        feature_list = []

        # Creating Vectors for every sentence. When a word is present, we give the value of 1 in its position in the vector. Else, we give the value of 0.
        for sentence_for_vector in texts:
            vector_list = []
            for word_for_vector in sentence_for_vector.split():
                if word_for_vector == sentence_for_vector.split()[0] and len(vector_list) < len(my_list):
                    for my_word in my_list:
                        if word_for_vector == my_word:
                            vector_list.append(1)
                        else:
                            vector_list.append(0)
                else:
                    for my_word in my_list:
                        if word_for_vector == my_word:
                            index = my_list.index(my_word)
                            vector_list[index] = 1
            feature = np.array(vector_list)
            feature_list.append(feature)

        vectors = np.array(feature_list, dtype=int)
        empty_list = []
        # Using the same method of creating vectors for the user's input
        for sentence in examined_texts:
            second_empty_list = []
            words = process_text(sentence)
            for item in words:
                if item == words[0] and len(empty_list) < len(my_list):
                    for my_word in my_list:
                        if item == my_word:
                            second_empty_list.append(1)
                        else:
                            second_empty_list.append(0)
                else:
                    for my_word in my_list:
                        if item == my_word:
                            index = my_list.index(my_word)
                            second_empty_list[index] = 1
                        else:
                            pass
            vector = np.array(second_empty_list)
            empty_list.append(vector)

        examined_vectors = np.array([empty_list])
        # Using the Multinomial Naive Bayes classifier the machine makes a satisfactory and accurate prediction about the topic of the sentence.
        classifier = MultinomialNB()
        classifier.fit(vectors, topics)
        another_list = []

        for examined_vector in examined_vectors:
            for exam_vect in examined_vector:
                topics_pred = classifier.predict(examined_vector)

        text_1 = tkinter.Text(top, font=("Helvetica", 24), wrap=tkinter.WORD)
        text_2 = tkinter.Label(top, text="You can also talk with others who have similar experiences in our chat room.",
                               font=("Helvetica", 24))
        room = 0
        text_1.insert(tkinter.END, feel_better_answer)

        text_1.pack()
        text_2.pack()

        for topic in topics_pred:
            if topic == "Family":
                room = 1
            elif topic == "Social":
                room = 2
            elif topic == "School":
                room = 3
            elif topic == "COVID-19":
                room = 4
            elif topic == "Bullying":
                room = 5

        my_button = tkinter.Button(top, text="Click Here", font=("Helvetica", 24), command=lambda: open_browser(room))
        my_button.pack()


button_for_start = tkinter.Button(top, text="Press here to Start", font=("Helvetica", 24), command=lambda: start())
button_for_start.place(anchor="center")
button_for_start.pack()

top.mainloop()
