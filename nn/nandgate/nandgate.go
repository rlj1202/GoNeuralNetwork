package main

import (
	"github.com/rlj1202/GoNeuralNetwork/nn"
	"github.com/rlj1202/GoNeuralNetwork/mat"
	"log"
)

func main() {
	trainingData := []nn.TrainingData{
		{mat.NewColVector(2, []float64{0, 0}, 0), mat.NewColVector(1, []float64{1}, 0)},
		{mat.NewColVector(2, []float64{0, 1}, 0), mat.NewColVector(1, []float64{1}, 0)},
		{mat.NewColVector(2, []float64{1, 0}, 0), mat.NewColVector(1, []float64{1}, 0)},
		{mat.NewColVector(2, []float64{1, 1}, 0), mat.NewColVector(1, []float64{0}, 0)},
	}

	net := nn.NewNetwork([]int{2, 1}, nil, nil)
	costs := net.StochasticGradientDescent(trainingData, 500, 4, 10.0, trainingData, nn.CE)

	nn.PlotCosts(25, 400.0, costs)

	for _, data := range trainingData {
		_, as := net.FeedForward(data.In)
		a := as[len(as) - 1]

		log.Printf("%.2v : %.2v (%.2v)\n", data.In.Data, a.Data, data.Out.Data)
	}

	for i, bias := range net.Biases {
		log.Printf("bias %d\n", i)
		log.Println(bias)
	}
	for i, weight := range net.Weights {
		log.Printf("weight %d\n", i)
		log.Println(weight)
	}

	nn.WriteCostsResult("./NandGate.csv", costs)
}
