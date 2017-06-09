package main

import (
	"math/rand"
	"time"
	"math"
	"log"
	"fmt"
)

type Network struct {
	NumLayers int `json:"num_layers"`
	Sizes     []int `json:"sizes"`
	Biases    []Matrix `json:"biases"`
	Weights   []Matrix `json:"weights"`
}
type TrainingData struct {
	in Matrix// col vector
	out Matrix// col vector
}

func NewNetwork(sizes []int, biases []Matrix, weights []Matrix) Network {
	rand.Seed(time.Now().Unix())
	random := func (a float64) float64 {
		return rand.Float64() * 4 - 2
	}

	numLayers := len(sizes)
	if biases == nil {
		biases = make([]Matrix, numLayers - 1)

		for i, size := range sizes[1:] {
			biases[i] = NewColVector(size, []float64{}).Apply(random)
		}
	}
	if weights == nil {
		weights = make([]Matrix, numLayers - 1)

		for i, size := range sizes[1:] {
			weights[i] = NewMatrix(size, sizes[i], []float64{}).Apply(random)
		}
	}

	return Network{numLayers, sizes, biases, weights}
}

// Returns slices of z and a in each layer. z is a weighted input and a is an activation.
func (net Network) FeedForward(a Matrix) ([]Matrix, []Matrix) {
	zs := make([]Matrix, net.NumLayers - 1)
	as := make([]Matrix, net.NumLayers - 1)

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
func (net Network) StochasticGradientDescent(trainingSet []TrainingData, epochs int, miniBatchSize int, eta float64, testSet []TrainingData) (costs []float64) {
	costs = make([]float64, 0)

	for epoch := 1; epoch <= epochs; epoch++ {
		log.Printf("Start epoch: %d\n", epoch)
		fmt.Printf("Start epoch: %d\n", epoch)
		// shuffle
		log.Println("	Shuffling training data")
		shuffle := make([]TrainingData, len(trainingSet))
		rand.Seed(time.Now().Unix())
		for i, value := range rand.Perm(len(trainingSet)) {
			shuffle[i] = trainingSet[value]
		}
		trainingSet = shuffle

		log.Println("	Training mini batch")
		for offset := 0; offset < len(trainingSet); offset += miniBatchSize {
			miniBatch := trainingSet[offset:offset+miniBatchSize]

			if offset % 1000 == 0 {
				log.Printf("		mini batch at %d\n", offset)
			}

			nablaB_sum := make([]Matrix, net.NumLayers - 1)
			nablaW_sum := make([]Matrix, net.NumLayers - 1)
			for i := 0; i < net.NumLayers - 1; i++ {
				nablaB_sum[i] = NewMatrix(net.Biases[i].Rows, net.Biases[i].Cols, nil)
				nablaW_sum[i] = NewMatrix(net.Weights[i].Rows, net.Weights[i].Cols, nil)
			}

			//log.Println("		Training each image")
			for _, x := range miniBatch {
				//log.Println("			Calculating gradient descent through BackPropagate")
				nablaB, nablaW := net.BackPropagate(x)

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
		cost_sum := 0.0
		differ_sum := NewColVector(net.Sizes[len(net.Sizes) - 1], nil)
		corrects := 0
		for _, testData := range testSet {
			_, as := net.FeedForward(testData.in)
			a := as[len(as) - 1]

			differ := testData.out.Sub(a).Apply(math.Abs)
			differ_sum = differ_sum.Add(differ)

			differ_inner := differ.Transpose().MatProd(differ)
			cost_sum += differ_inner.At(1, 1)

			log.Printf("actual output %.2v\n", a.Data)
			maxIndex := 0
			maxValue := 0.0
			for i, value := range a.Data {
				if value > maxValue {
					maxValue = value
					maxIndex = i
				}
			}

			log.Printf("expected output %.2v\n", testData.out.Data)
			numIndex := 0
			for i, value := range testData.out.Data {
				if value == 1.0 {
					numIndex = i
					break
				}
			}

			if maxIndex == numIndex {
				corrects++
			}
		}
		cost := cost_sum / float64(len(testSet))
		differVec := differ_sum.Apply(func (a float64) float64 { return a / float64(len(testSet)) })
		log.Printf("		cost: %f\n", cost)
		log.Printf("		differVec: %.2v\n", differVec.Data)
		log.Printf("		corrects: %d\n", corrects)
		log.Printf("		total test datas: %d\n", len(testSet))

		costs = append(costs, cost)

		_, temp := net.FeedForward(testSet[0].in)
		log.Printf("			in %.2v\n", testSet[0].in.Data)
		log.Printf("			out %.2v\n", temp[len(temp) - 1].Data)
		log.Printf("			expected %.2v\n", testSet[0].out.Data)

		fmt.Printf("	cost: %f\n", cost)
	}

	return
}

// Returns slices of nabla b and nabla w in each layer.
func (net Network) BackPropagate(trainingData TrainingData) ([]Matrix, []Matrix) {
	delta := make([]Matrix, net.NumLayers - 1)
	nablaB := make([]Matrix, net.NumLayers - 1)
	nablaW := make([]Matrix, net.NumLayers - 1)

	zs, as := net.FeedForward(trainingData.in)
	as = append([]Matrix{trainingData.in}, as...)
	a_L := as[len(as) - 1]
	z_L := zs[len(zs) - 1]
	costDerivative := a_L.Sub(trainingData.out)

	delta_L := costDerivative.Mul(z_L.Apply(sigmoidPrime))// equation (BP1)
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

func sigmoid(value float64) float64 {
	return 1.0 / (1.0 + math.Exp(-value))
}

func sigmoidPrime(value float64) float64 {
	return sigmoid(value) * (1 - sigmoid(value))
}
