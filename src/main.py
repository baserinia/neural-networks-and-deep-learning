#!/usr/bin/env python3

import mnist_loader
import network
import mnist_svm

def run_network():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 100, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


def main():
    run_network()

if __name__ == "__main__":
    main()
