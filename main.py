from shortcut import load_labeled_vectors
from networks.network import Network

image_number, row, col, dataset = load_labeled_vectors('training_dataset/train_images','training_dataset/train_labels')
image_number_, row_, col_, test_set = load_labeled_vectors('test_dataset/test_images','test_dataset/test_labels')
net = Network([row*col,64,10])


# net.SGD(dataset,30,.1,test_set,)
net.BGD(dataset,10000,1,test_set,)
# net.MGD(dataset,1000,3,10,test_set,)

# net.test(test_set)