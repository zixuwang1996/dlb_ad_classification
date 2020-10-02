import gensim
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import numpy as np
import pickle
import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_embedding', '-p', type=str, required=True)
    parser.add_argument('--npy_output', type=str, required=True)
    parser.add_argument('--dict_output', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dump_frequency', type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Vocabulary
    print("Building Vocabulary ...")
    dataset_vocab = []
    with open(args.dataset) as dfile:
        for line in dfile.readlines():
            dataset_vocab.extend(line.split())
    print("Vocablary size is :")
    print(len(set(dataset_vocab)))
#    print(set(dataset_vocab))
    data = {'': 0}
    embeddings = [np.zeros((200), dtype=np.float32)]
    float_re = re.compile(' [-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?')
 
    # 2. Embeddings
    print("Building embeddings ...")
    model = KeyedVectors.load_word2vec_format(args.pretrained_embedding, binary=True)
#    print(model.vocab)
    print("Pretrained embeddings loaded !")
   
    with open(args.dict_output, 'wb') as dfile, \
         open(args.npy_output, 'wb') as nfile:
 
        idx = 1
        for word in set(dataset_vocab):

            if word in model.vocab:
                embeddings.append(model[word])
                if word not in data:
                    data[word] = idx
                 
                idx += 1

            else:
                continue
                #embeddings.append(np.zeros(model.vector_size))

#            if not idx % args.dump_frequency:
#                np.save(nfile, np.asarray(embeddings))
#                embeddings.clear()

        np.save(nfile, np.asarray(embeddings))
        pickle.dump(data, dfile)

    print("Vocabulary saved, size is {} words".format(idx))

if __name__ == '__main__':
    main()







'''
with open("BioWordVec_PubMed_MIMICIII_d200.bin", "rb") as f:
    for line in f:
        print(line)
    
    vocab_size, layer1_size = map(int, header.split())   
    print(vocab_size)
    print(layer1_size)

    binary_len = np.dtype('float32').itemsize * layer1_size
    print(binary_len)
     
'''

