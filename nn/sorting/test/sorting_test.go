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

		inputs = append(inputs, mat.NewColVector(3, normDigits, 0))
	}

	corrects := 0
	for _, input := range inputs {
		result := loadNetworkAndFeedForward("E:/RedLaboratory/Go/Workspace/src/github.com/rlj1202/GoNeuralNetwork/net_SortThreeDigits_ce.json", input)

		a := make([]int, 3)
		for i, value := range input.Data {
			a[i] = int(value * 9.0)
		}

		sorted := []int{mat.Argmax(result.Data[0:10]), mat.Argmax(result.Data[10:20]), mat.Argmax(result.Data[20:30])}

		bubble := make([]int, 3)
		copy(bubble, sorted)

		for i := 0; i < 3; i++ {
			for j := 0; j < 2; j++ {
				if bubble[j] > bubble[j+1] {
					bubble[j], bubble[j+1] = bubble[j+1], bubble[j]
				}
			}
		}

		correct := true
		for i := 0; i < 3; i++ {
			if sorted[i] != bubble[i] {
				correct = false
			}
		}
		if correct { corrects++ }

		fmt.Printf("%.2v -> %.2v -> %.2v\n", a, input.Data, sorted)
	}

	fmt.Println(len(inputs), corrects)
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

