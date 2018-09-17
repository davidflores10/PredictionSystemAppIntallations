# Create your views here.


from django.http import HttpResponse
from django.shortcuts import render_to_response, render
from django.template import Context, loader
from numpy.ma import array

from System.neural_network import NeuralNetwork, NeuralEvenAndOdd


def index(request):
    return render(request, "index.html")


def neural(request):
    neuralNetwork = NeuralNetwork()

    # variables
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neuralNetwork.train(training_set_inputs, training_set_outputs, 1000)

    return HttpResponse(neuralNetwork.think(array([1, 0, 0])))


def neural_even_and_odd(request):
    neuralEvenAndOdd = NeuralEvenAndOdd()

    # variables
    training_set_inputs = array(
        [[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 1],
         [0, 0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1]])

    training_set_outputs = array([[1, 0, 1, 1, 0, 0, 1]]).T

    neuralEvenAndOdd.train(training_set_inputs, training_set_outputs, 10000)
    # return HttpResponse(neuralEvenAndOdd.think(array([0, 0, 0, 0, 0, 0, 1, 0])))
    return HttpResponse(neuralEvenAndOdd.think(convertToBinaryArray(2)))


def convertToBinaryArray(number):
    """
    Method for make array of 8 bits for a int number
    :param number:
    :return:
    """
    binary = bin(number)
    array = []
    length = 8
    i = 0

    if len(str(bin(number))[2:]) < 8:
        error_length = length - len(str(bin(number))[2:])
        while i < error_length:
            array.append(int(0))
            i += 1

    for d in str(bin(number))[2:]:
        array.append(int(d))

    return array
