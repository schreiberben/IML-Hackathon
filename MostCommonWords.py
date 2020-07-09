import collections
import pandas as pd
import matplotlib.pyplot as plt
import re
# Read input file, note the encoding is specified here
# It may be different in your text file

DELIMITERS = ",", ".", ")", "(", "/", "\n", "{", "}", "=", "'", ":", ",", "\"", "[", "]"
regexPattern = '|'.join(map(re.escape, DELIMITERS))

allPaths = ["building_tool_all_data.txt", "espnet_all_data.txt", "horovod_all_data.txt",
            "jina_all_data.txt", "PaddleHub_all_data.txt", "PySolFC_all_data.txt",
            "pytorch_geometric_all_data.txt"]


def list_most_common_words(path, amount):
    """
    Finds the most common words in the text
    :param path: path to txt file
    :param amount: the otp how many words to find in the text
    :return: a list of the most common words
    """
    global count
    file = open(path, encoding="utf8")
    a = file.read()

    # Stopwords

    stopwords = [""]

    # Instantiate a dictionary, and for every word in the file,
    # Add to the dictionary if it doesn't exist. If it does, increase the count.
    wordcount = {}
    # To eliminate duplicates, remember to split by punctuation, and use case demiliters.

    length_of_text = 0
    for word in re.split(regexPattern, a.lower()):
        word = word.replace(" ", "")

        if word not in stopwords:
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
        length_of_text += 1

    # Print most common words
    n_print = amount
    print("\nOK. The {} most common words are as follows\n".format(n_print))
    word_counter = collections.Counter(wordcount)
    for word, count in word_counter.most_common(n_print):
        print(word, ": ", count)
    # Close the file
    file.close()

    # Create a list of the most common words

    lst = word_counter.most_common(n_print)

    # Draw a bar chart
    df = pd.DataFrame(lst, columns=['Word', 'Count'])
    df.plot.bar(x='Word', y='Count')
    plt.title(path)
    plt.show()

    return [lst[i][0] for i in range(len(lst))]


# Putting all the common words together

common_words = []
for path in allPaths:
    common_words += list_most_common_words(path, 50)

print(common_words)
