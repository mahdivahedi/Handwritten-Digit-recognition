import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.ndimage.interpolation import shift

sigmoid = True
shift_mode = False
momentum_SGD = False


def sig(x):
    try:
        return 1 / (1 + math.exp(-x))
    except:
        if x[0] > 0:
            return 1
        else:
            return 0


def sig_deriv(x):
    return sig(x) * (1 - sig(x))


def Relu(x):
    if x > 0:
        return x
    else:
        return x * 0.001


def Relu_deriv(x):
    if x > 0:
        return 1
    else:
        return 0


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - tanh(x) ** 2


def activation_func(x):
    if sigmoid:
        return sig(x)
    else:
        return tanh(x)


def activation_func_deriv(x):
    if sigmoid:
        return sig_deriv(x)
    else:
        return tanh_deriv(x)


def RAI(fan_in, fan_out):
    V = np.random.randn(fan_out, fan_in + 1) * 0.6007 / fan_in ** 0.5
    for j in range(fan_out):
        k = np.random.randint(0, high=fan_in + 1)
        V[j, k] = np.random.beta(2, 1)
    W = V[:, :-1]
    b = np.reshape(V[:, -1], (fan_out, 1))
    return W.astype(np.float32), b.astype(np.float32)


# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


def cost(al, image_number):
    c = 0
    for neuron_num in range(10):
        c += (al[neuron_num] - train_set[image_number][1][neuron_num]) ** 2
    return c


def shift_image(image):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [0, 4], cval=0, mode="constant")
    return shifted_image.reshape([-1])


##########################################################################################
learning_rate = 1
epoch = 5
batch_size = 50

momentum = 0.9

# Reading The Train Set
train_images_file = open('train-images.idx3-ubyte', 'rb')
train_images_file.seek(4)
num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
train_images_file.seek(16)

train_labels_file = open('train-labels.idx1-ubyte', 'rb')
train_labels_file.seek(8)

train_set = []
for n in range(num_of_train_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

    label_value = int.from_bytes(train_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1

    train_set.append((image, label))

# Reading The Test Set
test_images_file = open('t10k-images.idx3-ubyte', 'rb')
test_images_file.seek(4)

test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
test_labels_file.seek(8)

num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
test_images_file.seek(16)

test_set = []
for n in range(num_of_test_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

    label_value = int.from_bytes(test_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1

    test_set.append((image, label))

# shifted = shift_image(test_set[0][0])
# # # Plotting an image
# show_image(shifted)
# plt.show()

# w1 = np.random.normal(0, 2 / 784, size=(16, 784))
# b1 = np.zeros((16, 1))
# w2, b2 = RAI(16, 16)
# w3, b3 = RAI(16, 10)

w1 = np.random.randn(16, 784)
w2 = np.random.randn(16, 16)
w3 = np.random.randn(10, 16)

b1 = np.zeros((16, 1))
b2 = np.zeros((16, 1))
b3 = np.zeros((10, 1))

average_cost_epochs = []

# Training Part
start_time = time.time()
for e in range(epoch):
    c = 0
    for batch_number in range(num_of_train_images // batch_size):
        grad_w1 = np.zeros((16, 784))
        grad_w2 = np.zeros((16, 16))
        grad_w3 = np.zeros((10, 16))

        v1 = np.zeros((16, 784))
        v2 = np.zeros((16, 16))
        v3 = np.zeros((10, 16))

        grad_b1 = np.zeros(16)
        grad_b2 = np.zeros(16)
        grad_b3 = np.zeros(10)

        grad_a2 = np.zeros(16)
        grad_a3 = np.zeros(16)
        grad_a4 = np.zeros(10)

        for i in range(batch_size):
            image_no = batch_number * batch_size + i
            a1 = train_set[image_no][0]
            a2 = np.zeros([16])
            a3 = np.zeros([16])
            a4 = np.zeros([10])

            z2 = w1.dot(a1)
            z2 = np.add(z2, b1)
            a2 = np.array([activation_func(xi) for xi in z2]).reshape(16, 1)

            z3 = w2.dot(a2)
            z3 = np.add(z3, b2)
            a3 = np.array([activation_func(xi) for xi in z3]).reshape(16, 1)

            z4 = w3.dot(a3)
            z4 = np.add(z4, b3)
            a4 = np.array([activation_func(xi) for xi in z4]).reshape(10, 1)

            # Learning Part

            c += cost(a4, image_no)

            grad_a4 = 2 * (a4 - train_set[image_no][1])

            tmp = np.array([activation_func_deriv(xi) for xi in z4]).reshape(10, 1)
            grad_w3 += (tmp * grad_a4).dot(np.transpose(a3))

            grad_b3 = grad_a4 * tmp

            # sex = grad_a3
            grad_a3 = np.transpose(w3).dot(grad_a4 * tmp)

            tmp = np.array([activation_func_deriv(xi) for xi in z3]).reshape(16, 1)
            grad_w2 += (tmp * grad_a3).dot(np.transpose(a2))

            grad_b2 = grad_a3 * tmp

            grad_a2 = np.transpose(w2).dot(grad_a3 * tmp)
            tmp = np.array([activation_func_deriv(xi) for xi in z3]).reshape(16, 1)

            grad_w1 += (tmp * grad_a2).dot(np.transpose(a1))
            grad_b1 = grad_a2 * tmp
        # # SGD Part
        for j in range(10):
            b3[j] -= grad_b3[j] * learning_rate / batch_size

        for j in range(16):
            b2[j] -= grad_b2[j] * learning_rate / batch_size

        for j in range(16):
            b1[j] -= grad_b1[j] * learning_rate / batch_size

        if momentum_SGD:
            for j in range(10):
                for k in range(16):
                    v3[j, k] = v3[j, k] * momentum + grad_w3[j, k] * learning_rate / batch_size
                    w3[j, k] -= v3[j, k]

            for j in range(16):
                for k in range(16):
                    v2[j, k] = v2[j, k] * momentum + grad_w2[j, k] * learning_rate / batch_size
                    w2[j, k] -= v2[j, k]

            for j in range(16):
                for k in range(784):
                    v1[j, k] = v1[j, k] * momentum + grad_w1[j, k] * learning_rate / batch_size
                    w1[j, k] -= v1[j, k]

        else:
            for j in range(10):
                for k in range(16):
                    w3[j, k] -= grad_w3[j, k] * learning_rate / batch_size

            for j in range(16):
                for k in range(16):
                    w2[j, k] -= grad_w2[j, k] * learning_rate / batch_size

            for j in range(16):
                for k in range(784):
                    w1[j, k] -= grad_w1[j, k] * learning_rate / batch_size

    average_cost_epochs.append(c / num_of_train_images)

print("--- %s seconds ---" % (time.time() - start_time))
plt.plot(average_cost_epochs)
plt.show()

# Test Part
right_nums = 0
accuracy = 0
t = num_of_test_images
for i in range(t):
    if shift_mode:
        myimage = shift_image(test_set[i][0])
    else:
        myimage = test_set[i][0]
    a1 = myimage
    z2 = w1.dot(a1)
    z2 = np.add(z2, b1)
    a2 = np.array([activation_func(xi) for xi in z2]).reshape(16, 1)

    z3 = w2.dot(a2)
    z3 = np.add(z3, b2)
    a3 = np.array([activation_func(xi) for xi in z3]).reshape(16, 1)

    z4 = w3.dot(a3)
    z4 = np.add(z4, b3)
    a4 = np.array([activation_func(xi) for xi in z4]).reshape(10, 1)

    recognised_num = np.argmax(a4)

    if test_set[i][1][recognised_num] == 1:
        right_nums += 1

accuracy = right_nums / t
print("accuracy = ", accuracy)
