package main

import (
	"github.com/rlj1202/GoNeuralNetwork/nn"
	"github.com/rlj1202/GoNeuralNetwork/mat"
)

func main() {
	trainingDatas :=[]nn.TrainingData{
		{mat.NewColVector(1, []float64{1}, 0), mat.NewColVector(1, []float64{0}, 0)},
	}

	biases := []mat.Matrix{
		mat.NewMatrix(1, 1, []float64{2.0}, 0),
	}
	weights := []mat.Matrix{
		mat.NewMatrix(1 ,1, []float64{2.0}, 0),
	}

	net := nn.NewNetwork([]int{1, 1}, biases, weights)

	costs := net.StochasticGradientDescent(trainingDatas, 300, 1, 0.05, trainingDatas, nn.CE)

	nn.PlotCosts(30, 100.0, costs)
}
