package nn

import (
	"math/rand"
	"time"
	"math"
	"log"
	"github.com/rlj1202/GoNeuralNetwork/mat"
)

type Network struct {
	NumLayers int `json:"num_layers"`
	Sizes     []int `json:"sizes"`
	Biases    []mat.Matrix `json:"biases"`
	Weights   []mat.Matrix `json:"weights"`
}
type TrainingData struct {
	In mat.Matrix// col vector
	Out mat.Matrix// col vector
}
type CostFunction struct {
	Fn func (network Network, tds []TrainingData) float64
	Delta func (z, a, y mat.Matrix) mat.Matrix
}

var (
	MSE = CostFunction{MeanSquaredError, MeanSquaredErrorDelta}
	CE = CostFunction{CrossEntropy, CrossEntropyDelta}
)

func NewNetwork(sizes []int, biases []mat.Matrix, weights []mat.Matrix) Network {
	rand.Seed(time.Now().Unix())
	random := func (a float64) float64 {
		return rand.Float64() * 4 - 2
	}

	numLayers := len(sizes)
	if biases == nil {
		biases = make([]mat.Matrix, numLayers - 1)

		for i, size := range sizes[1:] {
			biases[i] = mat.NewColVector(size, nil, 0).Apply(random)
		}
	}
	if weights == nil {
		weights = make([]mat.Matrix, numLayers - 1)

		for i, size := range sizes[1:] {
			weights[i] = mat.NewMatrix(size, sizes[i], nil, 0).Apply(random)
		}
	}

	return Network{numLayers, sizes, biases, weights}
}

// Returns slices of z and a in each layer. z is a weighted input and a is an activation.
func (net Network) FeedForward(a mat.Matrix) ([]mat.Matrix, []mat.Matrix) {
	zs := make([]mat.Matrix, net.NumLayers - 1)
	as := make([]mat.Matrix, net.NumLayers - 1)

	for i := 0; i < net.NumLayers - 1; i++ {
		b := net.Biases[i]
		w := net.Weights[i]

		z := w.MatProd(a).Add(b)
		a = z.Apply(sigmoid)

		zs[i] = z
		as[i] = a
	}

	return zs, as
}

// Execute stochastic gradient descent algorithm to learning network.
func (net Network) StochasticGradientDescent(trainingSet []TrainingData, epochs int, miniBatchSize int, eta float64, testSet []TrainingData, costFunction CostFunction) (costs []float64) {
	costs = make([]float64, 0)

	for epoch := 1; epoch <= epochs; epoch++ {
		log.Printf("Start epoch: %d\n", epoch)
		// shuffle
		log.Println("	Shuffling training data")
		shuffle := make([]TrainingData, len(trainingSet))
		rand.Seed(time.Now().Unix())
		for i, value := range rand.Perm(len(trainingSet)) {
			shuffle[i] = trainingSet[value]
		}
		trainingSet = shuffle

		log.Println("	Training mini batch")
		for offset := 0; offset + miniBatchSize <= len(trainingSet); offset += miniBatchSize {
			miniBatch := trainingSet[offset:offset+miniBatchSize]

			if offset % 1000 == 0 {
				log.Printf("		mini batch at %d\n", offset)
			}

			nablaB_sum := make([]mat.Matrix, net.NumLayers - 1)
			nablaW_sum := make([]mat.Matrix, net.NumLayers - 1)
			for i := 0; i < net.NumLayers - 1; i++ {
				nablaB_sum[i] = mat.NewMatrix(net.Biases[i].Rows, net.Biases[i].Cols, nil, 0)
				nablaW_sum[i] = mat.NewMatrix(net.Weights[i].Rows, net.Weights[i].Cols, nil, 0)
			}

			//log.Println("		Training each image")
			for _, x := range miniBatch {
				//log.Println("			Calculating gradient descent through BackPropagate")
				nablaB, nablaW := net.BackPropagate(x, costFunction)

				//log.Println("			Summing nabla b and w in each training data")
				for i, nablaB_l := range nablaB {
					nablaB_sum[i] = nablaB_sum[i].Add(nablaB_l)
				}
				for i, nablaW_l := range nablaW {
					nablaW_sum[i] = nablaW_sum[i].Add(nablaW_l)
				}
			}

			// apply the rule
			//log.Println("		Manipulating bias and weight")
			for i, nablaB_sum_l := range nablaB_sum {
				net.Biases[i] = net.Biases[i].Sub(nablaB_sum_l.Apply(func (nablaB float64) float64 { return nablaB * eta / float64(len(miniBatch)) }))
			}

			for i, nablaW_sum_l := range nablaW_sum {
				net.Weights[i] = net.Weights[i].Sub(nablaW_sum_l.Apply(func (nablaB float64) float64 { return nablaB * eta / float64(len(miniBatch)) }))
			}
		}

		log.Println("	Evaluating")
		corrects := 0
		for _, testData := range testSet {
			_, as := net.FeedForward(testData.In)
			a := as[len(as) - 1]

			//log.Printf("actual output %.2v\n", a.Data)
			//log.Printf("expected output %.2v\n", testData.Out.Data)

			if mat.Argmax(a.Data) == mat.Argmax(testData.Out.Data) {
				corrects++
			}
		}
		cost := costFunction.Fn(net, testSet)
		log.Printf("		cost: %f\n", cost)
		log.Printf("		corrects(based on argmax): %d\n", corrects)
		log.Printf("		total test datas: %d\n", len(testSet))

		costs = append(costs, cost)

		_, temp := net.FeedForward(testSet[0].In)
		log.Printf("			in %.2v\n", testSet[0].In.Data)
		log.Printf("			out %.2v\n", temp[len(temp) - 1].Data)
		log.Printf("			expected %.2v\n", testSet[0].Out.Data)
	}

	return
}

// Returns slices of nabla b and nabla w in each layer.
func (net Network) BackPropagate(trainingData TrainingData, costFunction CostFunction) ([]mat.Matrix, []mat.Matrix) {
	delta := make([]mat.Matrix, net.NumLayers - 1)
	nablaB := make([]mat.Matrix, net.NumLayers - 1)
	nablaW := make([]mat.Matrix, net.NumLayers - 1)

	zs, as := net.FeedForward(trainingData.In)
	as = append([]mat.Matrix{trainingData.In}, as...)
	a_L := as[len(as) - 1]
	z_L := zs[len(zs) - 1]

	delta_L := costFunction.Delta(z_L, a_L, trainingData.Out)// equation (BP1)
	nablaB_L := delta_L// equation (BP3)
	nablaW_L := delta_L.MatProd(as[len(as) - 2].Transpose())// equation (BP4)

	delta[len(delta) - 1] = delta_L
	nablaB[len(nablaB) - 1] = nablaB_L
	nablaW[len(nablaW) - 1] = nablaW_L

	for i := 0; i < len(delta) - 1; i++ {
		l := len(delta) - 2 - i

		delta_l := net.Weights[l + 1].Transpose().MatProd(delta[l + 1]).Mul(zs[l].Apply(sigmoidPrime))// equation (BP2)
		nablaB_l := delta_l// equation (BP3)
		nablaW_l := delta_l.MatProd(as[l].Transpose())// equation (BP4)

		delta[l] = delta_l
		nablaB[l] = nablaB_l
		nablaW[l] = nablaW_l
	}

	return nablaB, nablaW
}

func MeanSquaredError(network Network, tds []TrainingData) float64 {
	cost := 0.0

	for _, td := range tds {
		_, as := network.FeedForward(td.In)
		a := as[len(as) - 1]
		y := td.Out

		differVec := y.Sub(a)
		differInner := differVec.Transpose().MatProd(differVec)

		cost += differInner.Data[0] / 2
	}

	cost /= float64(len(tds))

	return cost
}

// Calculate delta of output layer.
func MeanSquaredErrorDelta(z, a, y mat.Matrix) mat.Matrix {
	return a.Sub(y).Mul(z.Apply(sigmoidPrime))
}

func CrossEntropy(network Network, tds []TrainingData) float64 {
	cost := 0.0

	for _, td := range tds {
		_, as := network.FeedForward(td.In)
		a := as[len(as) - 1]
		y := td.Out

		one := mat.NewColVector(len(y.Data), nil, 1)
		differVec := y.Mul(a.Apply(math.Log)).Add(one.Sub(y).Mul(one.Sub(a).Apply(math.Log)))
		differInner := differVec.Transpose().MatProd(one)

		cost += differInner.Data[0]
	}

	cost /= -float64(len(tds))

	return cost
}

// Calculate delta of output layer.
func CrossEntropyDelta(z, a, y mat.Matrix) mat.Matrix {
	return a.Sub(y)
}

func sigmoid(value float64) float64 {
	return 1.0 / (1.0 + math.Exp(-value))
}

func sigmoidPrime(value float64) float64 {
	return sigmoid(value) * (1 - sigmoid(value))
}
