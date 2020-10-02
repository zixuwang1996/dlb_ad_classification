import numpy as np
import re
import sys
import os
import pickle


def convert_file(filepath, word_dict):
    dataset = []
    with open(filepath) as ifile:
        for line in ifile.readlines():
            line = line.strip().replace(";", "").lower()
            
            string = line
            
            string = re.sub(r"lewy", "", string)
            string = re.sub(r"body", "", string)
            string = re.sub(r"bodies", "", string)
            string = re.sub(r"dlb", "", string)
            string = re.sub(r"lbd", "", string)
            string = re.sub(r"ad", "", string)
            
            string = re.sub(r"parkinson", "", string)
            string = re.sub(r"hallucinations", "", string)
            
            
            line1 = string
            dataset.append([word_dict.get(w, 0) for w in line1.split(' ')])

    return dataset


def discover_dataset(f, wdict):
    return convert_file(f, wdict)
  

def pad_dataset(dataset, maxlen):
    return np.array(
        [np.pad(r, (0, maxlen-len(r)), mode='constant') if len(r) < maxlen else np.array(r[-maxlen:])
         for r in dataset])


# Class for dataset related operations
class CRISDataset():
    def __init__(self, path, dict_path, maxlen=128, test=None):
#        pos_path = os.path.join(path, 'pos')
#        neg_path = os.path.join(path, 'neg')
        self.test = test

        with open(dict_path, 'rb') as dfile:
            wdict = pickle.load(dfile)
        
        if test:
            self.dataset = pad_dataset(discover_dataset(os.path.join(path, 'test.txt'), wdict), maxlen)
        else:
            self.pos_dataset = pad_dataset(discover_dataset(os.path.join(path, 'train.pos'), wdict), maxlen)#.astype('i')
            self.neg_dataset = pad_dataset(discover_dataset(os.path.join(path, 'train.neg'), wdict), maxlen)#.astype('i')

    def __len__(self):
        return len(self.pos_dataset) + len(self.neg_dataset)

    def get_example(self, i):
        is_neg = i >= len(self.pos_dataset)
        dataset = self.neg_dataset if is_neg else self.pos_dataset
        idx = i - len(self.pos_dataset) if is_neg else i
        label = [1, 0] if is_neg else [0, 1]
        
        print (type(dataset[idx]))
        return (dataset[idx], np.array(label, dtype=np.int32))
    
    def load(self):
        
        if self.test:
            return self.dataset
        dataset = np.concatenate((self.pos_dataset, self.neg_dataset))
        labels = []
        
        for idx in range (0, len(self.pos_dataset)):
            labels.append([0, 1])
        
        for idx in range (0, len(self.neg_dataset)):
            labels.append([1, 0])
        
        return dataset, np.array(labels, dtype=np.int32)


# Function for handling word embeddings
def load_embeddings(path, size, dimensions):
    
    embedding_matrix = np.zeros((size, dimensions), dtype=np.float32)

    # As embedding matrix could be quite big we 'stream' it into output file
    # chunk by chunk. One chunk shape could be [size // 10, dimensions].
    # So to load whole matrix we read the file until it's exhausted.
    size = os.stat(path).st_size
    with open(path, 'rb') as ifile:
        pos = 0
        idx = 0
        while pos < size:
            chunk = np.load(ifile)
            chunk_size = chunk.shape[0]
            embedding_matrix[idx:idx+chunk_size, :] = chunk
            idx += chunk_size
            pos = ifile.tell()
    return embedding_matrix



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    '''
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    '''
#    print(string)
    '''
    string = re.sub(r"lewy", "", string)
    string = re.sub(r"body", "", string)
    string = re.sub(r"bodies", "", string)
    string = re.sub(r" dlb ", "", string)
    string = re.sub(r" lbd ", "", string)
    string = re.sub(r" ad ", "", string)
    '''
    string = re.sub(r"parkinson", "", string)
    string = re.sub(r"hallucinations", "", string)
#    print(string)

    return string.strip().lower()

#string = "and no adported perceptual disturbances hallucinations the last lewy. she feels that the change of"
#clean_str(string)


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding="utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
#    print(positive_examples)
    negative_examples = list(open(negative_data_file, "r", encoding="utf-8").readlines())
#    negative_examples =list(open(negative_data_file,"rb").read().decode("utf-8", "ignore"))
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples

    x_text2 = []
    for sent in x_text:
        tmp = len(sent.split())
        if tmp < 5000:
            x_text2.append(sent)
        else:
            ss = " ".join(sent.split()[-5000:])
            x_text2.append(ss)

    x_text3 = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text3, y]


def load_data_and_labels_test(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples

    x_text2 = []
    for sent in x_text:
        tmp = len(sent.split())
        if tmp < 5000:
            x_text2.append(sent)
        else:
            ss = " ".join(sent.split()[-5000:])
            x_text2.append(ss)



    x_text3 = [sent.strip().lower() for sent in x_text2]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text3, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
