# Create your views here.


from django.http import HttpResponse
from django.shortcuts import render_to_response, render
from django.template import Context, loader
from numpy.ma import array

from System.neural_network import NeuralNetwork


def index(request):
    return render(request, "index.html")


def neural(request):
    neuralNetwork = NeuralNetwork()

    # variables
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neuralNetwork.train(training_set_inputs, training_set_outputs, 1000)

    return HttpResponse(neuralNetwork.think(array([1, 0, 0])))
