package nn

import (
	"encoding/json"
	"io/ioutil"
	"strconv"
	"os"
	"encoding/csv"
	"bytes"
	"fmt"
	"log"
	"io"
)

func SaveNetwork(fileName string, network *Network) {
	jsonByte, err := json.Marshal(network)
	if err != nil {
		panic(err)
	}

	err = ioutil.WriteFile(fileName, jsonByte, 0644)
	if err != nil {
		panic(err)
	}
}

func LoadNetwork(fileName string) (*Network, error) {
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

func PlotCosts(lines int, amp float64, costs []float64) {
	for i := 0; i < lines; i++ {
		index := int(float64(len(costs)) * (float64(i) / float64(lines)))
		cost := costs[index]
		buffer := bytes.NewBufferString("")
		buffer.WriteString(fmt.Sprintf("%04v ", index))
		for j := 0; j < int(cost * amp); j++ {
			buffer.WriteString(fmt.Sprint("#"))
		}
		buffer.WriteString(fmt.Sprintf("%.2v\n", cost))
		log.Print(buffer.String())
	}
}

func ConcatCostsResult(fileNames ...string) []float64 {
	result := make([]float64, 0)

	for _, fileName := range fileNames {
		costs := ReadCostsResult(fileName)

		result = append(result, costs...)
	}

	return result
}

func ReadCostsResult(fileName string) []float64 {
	result := make([]float64, 0)

	file, err := os.Open(fileName)
	defer file.Close()

	if err != nil {
		panic(err)
	}

	reader := csv.NewReader(file)

	record, err := reader.Read()
	if len(record) != 2 || record[0] != "epoch" || record[1] != "cost" {
		panic("not a epoch-cost csv file: " + record[0] + "," + record[1])
	}

	for {
		if record, err = reader.Read(); err != nil {
			if err == io.EOF {
				break
			} else {
				panic(err)
			}
		}

		cost, err := strconv.ParseFloat(record[1], 64)

		if err != nil {
			panic(err)
		}

		result = append(result, cost)
	}

	return result
}

func WriteCostsResult(fileName string, costs []float64) {
	file, err := os.OpenFile(fileName, os.O_WRONLY | os.O_CREATE | os.O_APPEND, 0666)

	defer file.Close()

	if err != nil {
		panic(err)
	}

	writer := csv.NewWriter(file)
	writer.Write([]string{"epoch", "cost"})
	for epoch, cost := range costs {
		writer.Write([]string{strconv.FormatInt(int64(epoch), 10), strconv.FormatFloat(cost, 'f', -1, 64)})
	}
	writer.Flush()
	err = writer.Error()
	if err != nil {
		panic(err)
	}
}
