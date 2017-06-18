package nn

import (
	"testing"
	"github.com/rlj1202/GoNeuralNetwork/mat"
)

func TestNetwork(t *testing.T) {
	m := mat.NewMatrix(2, 2, []float64{1.0, 2.0, 3.0, 4.0})
	t.Log(m)
}
