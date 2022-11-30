def load_data():
    """ Specifies the used dataset, currently MNIST.

    :return: Four np arrays, namely x_train, y_train, x_test, y_test
    """
    from keras.datasets.mnist import load_data as dataset

    # f=np.load("/work/msimklue/pap/data.npz")
    # return (f["train_x"],f["train_y"]),(f["test_x"],f["test_y"])

    return dataset()


# If executed directly
if __name__ == "__main__":
    (x, y), (tx, ty) = load_data()

    print(x.shape)
