package nn

import (
	"testing"
	"github.com/rlj1202/GoNeuralNetwork/mat"
	"math"
)

func TestNetwork(t *testing.T) {
	network := NewNetwork([]int{2, 2, 2}, nil, nil)

	tds := []TrainingData{
		{mat.NewColVector(2, []float64{0.5, 0.5}, 0), mat.NewColVector(2, []float64{0.5, 0.5}, 0)},
	}

	cost := 0.0

	for _, td := range tds {
		_, as := network.FeedForward(td.In)
		a := as[len(as) - 1]
		y := td.Out

		one := mat.NewColVector(len(y.Data), nil, 1.0)

		t.Log(a, y, one)

		differVec := y.Mul(a.Apply(math.Log)).Add(one.Sub(y).Mul(one.Sub(a).Apply(math.Log)))
		differInner := differVec.Transpose().MatProd(one)

		cost += differInner.Data[0]
	}

	cost /= -float64(len(tds))

	t.Log(cost)
}
