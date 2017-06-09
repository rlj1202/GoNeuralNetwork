package main

import (
	"os"
	"io/ioutil"
	"strconv"
	"log"
	"bytes"
	"fmt"
	"math/rand"
	"time"
	"math"
	"encoding/json"
)

func main() {
	logFileName := "log_"
	logFileName = string(append([]byte(logFileName), []byte(time.Now().Format("2006-01-02_03-04"))...))
	logFileName = string(append([]byte(logFileName), []byte(".txt")...))
	logFile, err := os.OpenFile(logFileName, os.O_CREATE | os.O_WRONLY | os.O_APPEND, 0666)
	if err != nil {
		panic(err)
	}
	defer logFile.Close()

	log.SetOutput(logFile)

	//singleNeuron()
	//logicNANDGate()
	//handWriting()
	//sorting()

	inputs := make([]Matrix, 0)
	for i := 0; i < 100; i++ {
		digits := rand.Perm(10)[:3]
		//for j := 0; j < 2; j++ {
		//	for k := 0; k < 2; k++ {
		//		if digits[k] > digits[k+1] {
		//			digits[k+1], digits[k] = digits[k], digits[k+1]
		//		}
		//	}
		//}

		normDigits := make([]float64, 3)
		for j, digit := range digits {
			normDigits[j] = float64(digit) / 9.0
		}

		inputs = append(inputs, NewColVector(3, normDigits))
	}

	for _, input := range inputs {
		result := loadNetworkAndFeedForward("./net_SortThreeDigits.json", input)

		a := make([]int, 3)
		for i, value := range input.Data {
			a[i] = int(value * 9.0)
		}

		fmt.Printf("%.2v -> %.2v -> %.2v\n", a, input.Data, []int{argmax(result.Data[0:10]), argmax(result.Data[10:20]), argmax(result.Data[20:30])})
	}
}

func loadNetworkAndFeedForward(fileName string, input Matrix) Matrix {
	network, err := loadNetwork(fileName)
	if err != nil {
		panic(err)
	}
	_, as := network.FeedForward(input)
	result := as[len(as) - 1]

	return result
}

func sorting() {
	trainingData := []TrainingData{
		//{NewColVector(3, []float64{0.3, 0.6, 0.9}), NewColVector(3, []float64{0.3, 0.6, 0.9})},
		//{NewColVector(3, []float64{0.3, 0.9, 0.6}), NewColVector(3, []float64{0.3, 0.6, 0.9})},
		//{NewColVector(3, []float64{0.6, 0.3, 0.9}), NewColVector(3, []float64{0.3, 0.6, 0.9})},
		//{NewColVector(3, []float64{0.6, 0.9, 0.3}), NewColVector(3, []float64{0.3, 0.6, 0.9})},
		//{NewColVector(3, []float64{0.9, 0.3, 0.6}), NewColVector(3, []float64{0.3, 0.6, 0.9})},
		//{NewColVector(3, []float64{0.9, 0.6, 0.3}), NewColVector(3, []float64{0.3, 0.6, 0.9})},
	}

	generateTrainingData := func (sets int) []TrainingData {
		data := make([]TrainingData, 0)

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

			for _, perm := range permutations(digits) {
				td := TrainingData{NewColVector(len(digits), toFloat(perm)), NewColVector(len(digits)*10, toOneHotVec(digits))}
				data = append(data, td)
			}
		}

		return data
	}

	trainingData = generateTrainingData(300)
	testData := generateTrainingData(5)

	log.Printf("trainingDatas: %v\n", len(trainingData))
	log.Printf("testDatas: %v\n", len(testData))

	network, _ := loadNetwork("./net_SortThreeDigits.json")
	if network == nil {
		*network = NewNetwork([]int{3, 40, 40, 30}, nil, nil)
	}
	costs := network.StochasticGradientDescent(trainingData, 1000, 6, 0.1, testData)

	plotCosts(100, costs)

	for _, data := range trainingData {
		_, as := network.FeedForward(data.in)
		a := as[len(as) - 1]

		log.Printf("%.2v : %.2v (%.2v)\n", data.in.Data, a.Data, data.out.Data)
	}

	for i, bias := range network.Biases {
		log.Printf("bias %d\n", i)
		printMatrix(bias)
	}
	for i, weight := range network.Weights {
		log.Printf("weight %d\n", i)
		printMatrix(weight)
	}

	for _, test := range testData {
		_, as := network.FeedForward(test.in)
		result := as[len(as) - 1]

		convertToInt := func (a []float64) []int {
			r := make([]int, len(a))

			for i, value := range a {
				r[i] = int(math.Floor(value * 9.0 + 0.5))
			}

			return r
		}

		log.Printf("int: %.2v -> result: %.2v | out: %.2v\n", convertToInt(test.in.Data), [][]float64{softmax(result.Data[0:10]), softmax(result.Data[10:20]), softmax(result.Data[20:30])}, test.out.Data)
	}

	writeCostsResult("./SortThreeDigits.csv", costs)
	saveNetwork("./net_SortThreeDigits.json", network)
}

func singleNeuron() {
	trainingDatas :=[]TrainingData{
		{NewColVector(1, []float64{1}), NewColVector(1, []float64{0})},
	}

	biases := []Matrix{
		NewMatrix(1, 1, []float64{2.0}),
	}
	weights := []Matrix{
		NewMatrix(1 ,1, []float64{2.0}),
	}

	net := NewNetwork([]int{1, 1}, biases, weights)

	costs := net.StochasticGradientDescent(trainingDatas, 300, 1, 0.15, trainingDatas)

	plotCosts(25, costs)
}

func handWriting() {
	//testA := NewRowVector(10, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 0})
	//testB := NewRowVector(10, []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
	//testA = testA.Add(testB)
	//printMatrix(testA)
	//printMatrix(testB)

	trainImageFile, _ := os.Open("mnist/train-images.idx3-ubyte")
	trainLabelFile, _ := os.Open("mnist/train-labels.idx1-ubyte")
	trainingDataSet := getTrainingData(trainImageFile, trainLabelFile)

	testImageFile, _ := os.Open("mnist/t10k-images.idx3-ubyte")
	testLabelFile, _ := os.Open("mnist/t10k-labels.idx1-ubyte")
	testDataSet := getTrainingData(testImageFile, testLabelFile)

	printMatrix(Matrix{28, 28, testDataSet[0].in.Data}.Apply(func (a float64) float64 {
		//if a > 0.5 {
		//	return 1.0
		//} else {
		//	return 0.0
		//}
		return a
	}))
	printMatrix(Matrix{1, 10, testDataSet[0].out.Data})

	net := NewNetwork([]int{784, 30, 10}, nil, nil)

	for i, bias := range net.Biases {
		log.Printf("layer %d bias {%d, %d}\n", i + 2, bias.Rows, bias.Cols)
	}
	for i, weight := range net.Weights {
		log.Printf("layer %d weight {%d, %d}\n", i + 2, weight.Rows, weight.Cols)
	}

	net.StochasticGradientDescent(trainingDataSet, 30, 10, 5.0, testDataSet[:10])
}

func logicNANDGate() {
	trainingData := []TrainingData{
		{NewColVector(2, []float64{0, 0}), NewColVector(1, []float64{1})},
		{NewColVector(2, []float64{0, 1}), NewColVector(1, []float64{1})},
		{NewColVector(2, []float64{1, 0}), NewColVector(1, []float64{1})},
		{NewColVector(2, []float64{1, 1}), NewColVector(1, []float64{0})},
	}

	net := NewNetwork([]int{2, 1}, nil, nil)
	costs := net.StochasticGradientDescent(trainingData, 500, 4, 3.0, trainingData)

	plotCosts(25, costs)

	for _, data := range trainingData {
		_, as := net.FeedForward(data.in)
		a := as[len(as) - 1]

		log.Printf("%.2v : %.2v (%.2v)\n", data.in.Data, a.Data, data.out.Data)
	}

	for i, bias := range net.Biases {
		log.Printf("bias %d\n", i)
		printMatrix(bias)
	}
	for i, weight := range net.Weights {
		log.Printf("weight %d\n", i)
		printMatrix(weight)
	}

	writeCostsResult("./NandGate.csv", costs)
}

// mnist
func getTrainingData(images *os.File, labels *os.File) []TrainingData {
	imageIdx := ReadIDX(images)
	labelIdx := ReadIDX(labels)

	result := make([]TrainingData, imageIdx.GetSizeInDimension(0))
	for i := 0; i < len(result); i++ {
		imageBytes := imageIdx.Get(i)
		label := int(labelIdx.Get(i)[0])

		imageVector := NewColVector(784, nil)
		for i, b := range imageBytes {
			imageVector.Set(1, i + 1, float64(b) / 255.0)
		}
		labelVector := NewColVector(10, nil)
		labelVector.Set(1, label + 1, 1)

		result[i] = TrainingData{imageVector, labelVector}
	}

	return result
}

func printMatrix(mat Matrix) {
	buffer := bytes.NewBufferString("")

	for i := 0; i < mat.Cols * 5 + 3; i++ { buffer.WriteString(fmt.Sprint("-")) }
	log.Print(buffer.String());buffer.Reset()
	for row := 1; row <= mat.Rows; row++ {
		buffer.WriteString(fmt.Sprint("| "))
		for col := 1; col <= mat.Cols; col++ {
			buffer.WriteString(fmt.Sprintf("%#-4.2v ", mat.At(row, col)))
		}
		buffer.WriteString(fmt.Sprintln("|"))
	}
	log.Print(buffer.String());buffer.Reset()
	for i := 0; i < mat.Cols * 5 + 3; i++ { buffer.WriteString(fmt.Sprint("-")) }
	log.Print(buffer.String());buffer.Reset()
}

func plotCosts(lines int, costs []float64) {
	for i := 0; i < lines; i++ {
		index := int(float64(len(costs)) * (float64(i) / float64(lines)))
		cost := costs[index]
		buffer := bytes.NewBufferString("")
		buffer.WriteString(fmt.Sprintf("%04v ", index))
		for j := 0; j < int(cost*200.0); j++ {
			buffer.WriteString(fmt.Sprint("#"))
		}
		buffer.WriteString(fmt.Sprintf("%.2v\n", cost))
		log.Print(buffer.String())
	}
}

func writeCostsResult(fileName string, costs []float64) {
	data := make([]byte, 0)

	data = append(data, []byte("epoch, cost")...)

	for epoch, cost := range costs {
		data = append(data, []byte("\n")...)
		data = append(data, []byte(strconv.FormatInt(int64(epoch), 10))...)
		data = append(data, []byte(", ")...)
		data = append(data, []byte(strconv.FormatFloat(cost, 'f', -1, 64))...)
	}

	err := ioutil.WriteFile(fileName, data, 0644)
	log.Println("Write costs per epoch to file.")
	log.Println(err)
}

func permutations(input []int)[][]int{
	arr := make([]int, len(input))
	for i, value := range input {
		arr[i] = value
	}

	var helper func([]int, int)
	res := [][]int{}

	helper = func(arr []int, n int){
		if n == 1{
			tmp := make([]int, len(arr))
			copy(tmp, arr)
			res = append(res, tmp)
		} else {
			for i := 0; i < n; i++{
				helper(arr, n - 1)
				if n % 2 == 1{
					tmp := arr[i]
					arr[i] = arr[n - 1]
					arr[n - 1] = tmp
				} else {
					tmp := arr[0]
					arr[0] = arr[n - 1]
					arr[n - 1] = tmp
				}
			}
		}
	}
	helper(arr, len(arr))
	return res
}

func saveNetwork(fileName string, network *Network) {
	jsonByte, err := json.Marshal(network)
	if err != nil {
		panic(err)
	}

	fmt.Println(string(jsonByte))

	err = ioutil.WriteFile(fileName, jsonByte, 0644)
	if err != nil {
		panic(err)
	}
}

func loadNetwork(fileName string) (*Network, error) {
	jsonByte, err := ioutil.ReadFile(fileName)
	if err != nil {
		return nil, err
	}

	var network Network

	err = json.Unmarshal(jsonByte, &network)
	if err != nil {
		return nil, err
	}

	return &network, nil
}

func argmax (a []float64) int {
	index := 0
	max := 0.0

	for i, value := range a {
		if value > max {
			index = i
			max = value
		}
	}

	return index
}

func softmax (a []float64) []float64 {
	r := make([]float64, len(a))

	sum := 0.0
	for _, value := range a {
		sum += math.Exp(value)
	}

	for i, value := range a {
		r[i] = math.Exp(value) / sum
	}

	return r
}
