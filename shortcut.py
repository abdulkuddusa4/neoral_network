import math
import numpy as np


VECTOR_SIGMOID = lambda x: 1/(1+np.exp(-x))


def print(*args,**kwargs):
    kwargs['end'] = '\n\n'
    __builtins__['print'](*args,**kwargs)


def bytes_int(bts):
    bit_st=''
    for i in bts:
        bit_st+=bin(i)[2:]
    return int(bit_st,base=2)


def load_image(file_path):
    with open(file_path,'rb') as file:
        magic_number = bytes_int(file.read(4))
        image_number = bytes_int(file.read(4))
        row_no = bytes_int(file.read(4))
        col_no = bytes_int(file.read(4))
        image = file.read(row_no*col_no)
        input_vectors = []
        while len(image)>0:
            input_vectors.append(np.array([[i/255] for i in image]))
            image = file.read(row_no*col_no)
        return image_number,row_no,col_no,input_vectors


def load_labels(file_path):
    with open(file_path,'rb') as file:
        magic_number = bytes_int(file.read(4))
        image_number = bytes_int(file.read(4))

        labels = []
        byte = file.read(1)
        while len(byte)>0:
            labels.append(np.array([[1.0] if ord(byte) == i else [0] for i in range(10)]))
            byte = file.read(1)
        return image_number, labels


def load_labeled_vectors(data_path,label_path):
    image_number, row,col,vectors = load_image(data_path)
    lno,labels = load_labels(label_path)

    return image_number, row,col, [(vector,label) for vector,label in zip(vectors, labels)]

