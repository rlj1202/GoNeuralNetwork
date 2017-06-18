package sorting_test

import (
	"testing"
	"fmt"
	"math/rand"
	"github.com/rlj1202/GoNeuralNetwork/mat"
	"github.com/rlj1202/GoNeuralNetwork/nn"
)

func TestSorting(t *testing.T) {
	inputs := make([]mat.Matrix, 0)
	for i := 0; i < 100; i++ {
		digits := rand.Perm(10)[:3]

		normDigits := make([]float64, 3)
		for j, digit := range digits {
			normDigits[j] = float64(digit) / 9.0
		}

		inputs = append(inputs, mat.NewColVector(3, normDigits))
	}

	for _, input := range inputs {
		result := loadNetworkAndFeedForward("./net_SortThreeDigits.json", input)

		a := make([]int, 3)
		for i, value := range input.Data {
			a[i] = int(value * 9.0)
		}

		fmt.Printf("%.2v -> %.2v -> %.2v\n", a, input.Data, []int{mat.Argmax(result.Data[0:10]), mat.Argmax(result.Data[10:20]), mat.Argmax(result.Data[20:30])})
	}
}

func loadNetworkAndFeedForward(fileName string, input mat.Matrix) mat.Matrix {
	network, err := nn.LoadNetwork(fileName)
	if err != nil {
		panic(err)
	}
	_, as := network.FeedForward(input)
	result := as[len(as) - 1]

	return result
}

