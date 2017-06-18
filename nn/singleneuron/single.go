package main

import (
	"github.com/rlj1202/GoNeuralNetwork/nn"
	"github.com/rlj1202/GoNeuralNetwork/mat"
)

func main() {
	trainingDatas :=[]nn.TrainingData{
		{mat.NewColVector(1, []float64{1}), mat.NewColVector(1, []float64{0})},
	}

	biases := []mat.Matrix{
		mat.NewMatrix(1, 1, []float64{2.0}),
	}
	weights := []mat.Matrix{
		mat.NewMatrix(1 ,1, []float64{2.0}),
	}

	net := nn.NewNetwork([]int{1, 1}, biases, weights)

	costs := net.StochasticGradientDescent(trainingDatas, 300, 1, 0.15, trainingDatas)

	nn.PlotCosts(25, 100.0, costs)
}
