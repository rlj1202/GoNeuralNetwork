package main

import (
	"log"
	"math/rand"
	"github.com/rlj1202/GoNeuralNetwork/nn"
	"time"
	"github.com/rlj1202/GoNeuralNetwork/mat"
	"math"
)

func main() {
	trainingData := []nn.TrainingData{}

	generateTrainingData := func (sets int) []nn.TrainingData {
		data := make([]nn.TrainingData, 0)

		rand.Seed(time.Now().Unix())
		for i := 0; i < sets; i++ {// generate training data
			digits := rand.Perm(10)[0:3]
			for j := 0; j < len(digits) - 1; j++ {// sort digits
				for k := 0; k < len(digits) - 1; k++ {
					if digits[k] > digits[k+1] {
						digits[k], digits[k+1] = digits[k+1], digits[k]
					}
				}
			}

			toFloat := func (a []int) []float64 {
				result := make([]float64, len(a))
				for i, digit := range a {
					result[i] = float64(digit) / 9.0
				}
				return result
			}

			toOneHotVec := func (a []int) []float64 {
				result := make([]float64, len(a)*10)

				for i, value := range a {
					result[i*10 + value] = 1
				}

				return result
			}

			for _, perm := range mat.Permutations(digits) {
				td := nn.TrainingData{In: mat.NewColVector(len(digits), toFloat(perm), 0), Out: mat.NewColVector(len(digits)*10, toOneHotVec(digits), 0)}
				data = append(data, td)
			}
		}

		return data
	}

	trainingData = generateTrainingData(300)
	testData := generateTrainingData(5)

	log.Printf("trainingDatas: %v\n", len(trainingData))
	log.Printf("testDatas: %v\n", len(testData))

	networkFileName := "./net_SortThreeDigits_ce.json"

	network, _ := nn.LoadNetwork(networkFileName)
	if network == nil {
		network = new(nn.Network)
		*network = nn.NewNetwork([]int{3, 40, 40, 30}, nil, nil)
	}
	costs := network.StochasticGradientDescent(trainingData, 100, 6, 0.05, testData, nn.CE)

	nn.PlotCosts(100, 200.0, costs)

	//for _, data := range trainingData {
	//	_, as := network.FeedForward(data.In)
	//	a := as[len(as) - 1]
	//
	//	log.Printf("%.2v : %.2v (%.2v)\n", data.In.Data, a.Data, data.Out.Data)
	//}

	//for i, bias := range network.Biases {
	//	log.Printf("bias %d\n", i)
	//	log.Println(bias)
	//}
	//for i, weight := range network.Weights {
	//	log.Printf("weight %d\n", i)
	//	log.Println(weight)
	//}

	for _, test := range testData {
		_, as := network.FeedForward(test.In)
		result := as[len(as) - 1]

		convertToInt := func (a []float64) []int {
			r := make([]int, len(a))

			for i, value := range a {
				r[i] = int(math.Floor(value * 9.0 + 0.5))
			}

			return r
		}

		log.Printf("int: %.2v -> result: %.2v | out: %.2v\n", convertToInt(test.In.Data), [][]float64{mat.Softmax(result.Data[0:10]), mat.Softmax(result.Data[10:20]), mat.Softmax(result.Data[20:30])}, test.Out.Data)
	}

	//nn.WriteCostsResult("./SortThreeDigits.csv", costs)
	nn.SaveNetwork(networkFileName, network)
}
