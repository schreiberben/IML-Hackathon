import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def process_files(file_name, index):
    """
    Take in the text file with the project and the index at which it should be placed.
    Then output a dataframe where the first column contains each line of the project and the second column
    is the index
    :param file_name: A string with the path to the file
    :param index: The index of the project
    :return: A Dataframe as described above
    """
    # Read the lines from the file and strip end of line character
    file = open(file_name + '_all_data.txt', 'r', encoding="latin1")
    lines = [line.rstrip() for line in file.readlines()]
    file.close()

    # To avoid overhead process the lines in batches
    n = len(lines)
    num_chunks = n // 10

    # Allocate the memory for the array
    dataset = np.empty((n, 1), dtype=np.dtype('U100'))
    for i in range(0, n, num_chunks):
        j = i + num_chunks
        size = min(j - i, n - i)
        temp = np.asarray(lines[i:j], dtype=np.unicode).reshape((size, 1))
        dataset[i:j] = temp

    # Make the label column
    class_label = np.full((n, 1), index)

    return pd.DataFrame(np.hstack([dataset, class_label]))


def split_dataset(dataset):
    """
    Splits the given dataset into training and testing subsets
    """
    class_label = dataset[:, -1]  # for last column
    dataset = dataset[:, :-1]  # for all but last column
    return train_test_split(dataset, class_label)


if __name__ == '__main__':
    project_list = ['building_tool', 'espnet', 'horovod', 'jina', 'PaddleHub', 'PySolFC', 'pytorch_geometric']
    df = pd.concat([process_files(p, index) for index, p in enumerate(project_list)], ignore_index=True)
    df.to_csv('design_mat.csv', index=False)  # Save all of the lines of every project to a design matrix and attach the correct labels
