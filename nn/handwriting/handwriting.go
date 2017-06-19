package main

import (
	"os"
	"github.com/rlj1202/GoNeuralNetwork/nn"
	"fmt"
	"time"
	"github.com/rlj1202/GoNeuralNetwork/mat"
	"github.com/rlj1202/GoNeuralNetwork/idx"
)

func main() {
	//testA := NewRowVector(10, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 0})
	//testB := NewRowVector(10, []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
	//testA = testA.Add(testB)
	//printMatrix(testA)
	//printMatrix(testB)

	trainImageFile, _ := os.Open("./mnist/train-images.idx3-ubyte")
	trainLabelFile, _ := os.Open("./mnist/train-labels.idx1-ubyte")
	trainingDataSet := getTrainingData(trainImageFile, trainLabelFile)

	testImageFile, _ := os.Open("./mnist/t10k-images.idx3-ubyte")
	testLabelFile, _ := os.Open("./mnist/t10k-labels.idx1-ubyte")
	testDataSet := getTrainingData(testImageFile, testLabelFile)

	//printMatrix(Matrix{28, 28, testDataSet[0].in.Data}.Apply(func (a float64) float64 {
	//	//if a > 0.5 {
	//	//	return 1.0
	//	//} else {
	//	//	return 0.0
	//	//}
	//	return a
	//}))
	//printMatrix(Matrix{1, 10, testDataSet[0].out.Data})

	network, _ := nn.LoadNetwork("./net_HandWriting.json")
	if network == nil {
		network = new(nn.Network)
		*network = nn.NewNetwork([]int{784, 30, 10}, nil, nil)
	}

	//for i, bias := range net.Biases {
	//	log.Printf("layer %d bias {%d, %d}\n", i + 2, bias.Rows, bias.Cols)
	//}
	//for i, weight := range net.Weights {
	//	log.Printf("layer %d weight {%d, %d}\n", i + 2, weight.Rows, weight.Cols)
	//}

	costs := network.StochasticGradientDescent(trainingDataSet, 2000, 10, 0.1, testDataSet[:10], nn.MSE)

	nn.PlotCosts(50, 200.0, costs)

	nn.SaveNetwork("./net_HandWriting.json", network)
	nn.WriteCostsResult(fmt.Sprintf("./HandWriting_%s.csv", time.Now().Format("2006-01-02_03-04")), costs)
}

// mnist
func getTrainingData(images *os.File, labels *os.File) []nn.TrainingData {
	imageIdx := idx.ReadIDX(images)
	labelIdx := idx.ReadIDX(labels)

	result := make([]nn.TrainingData, imageIdx.GetSizeInDimension(0))
	for i := 0; i < len(result); i++ {
		imageBytes := imageIdx.Get(i)
		label := int(labelIdx.Get(i)[0])

		imageVector := mat.NewColVector(784, nil, 0)
		for i, b := range imageBytes {
			imageVector.Set(1, i + 1, float64(b) / 255.0)
		}
		labelVector := mat.NewColVector(10, nil, 0)
		labelVector.Set(1, label + 1, 1)

		result[i] = nn.TrainingData{In: imageVector, Out: labelVector}
	}

	return result
}
