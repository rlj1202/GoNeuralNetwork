package main

import (
	"github.com/rlj1202/GoNeuralNetwork/nn"
	"github.com/rlj1202/GoNeuralNetwork/kec"
	"time"
	"github.com/rlj1202/GoNeuralNetwork/mat"
	"fmt"
	"bytes"
	"log"
)

func main() {
	id := "0010CZS440"

	speeds := kec.GetVehicleSpeeds(id)

	tds := make([]nn.TrainingData, 0)

	for dateStr, speed := range speeds {
		date, err := time.Parse("2006010215", dateStr)

		if err != nil {
			panic(err)
		}

		year := float64(date.Year() - 2010) / 5.0
		month := float64(date.Month() - 1) / 11.0
		day := float64(date.Day() - 1) / 30.0
		hour := float64(date.Hour()) / 23.0

		in := mat.NewColVector(4, []float64{year, month, day, hour})
		out := mat.NewColVector(1, []float64{speed / 150.0})
		td := nn.TrainingData{In: in, Out: out}
		tds = append(tds, td)
	}

	network, _ := nn.LoadNetwork("./net_VehicleSpeed.json")
	if network == nil {
		*network = nn.NewNetwork([]int{4, 40, 40, 1}, nil, nil)
	}
	//costs := network.StochasticGradientDescent(tds, 100, 50, 0.025, tds[:10])
	//plotCosts(30, 2000, costs)

	for hour := 0; hour < 24; hour++ {
		dateStr := time.Date(2015, 1, 3, hour, 0, 0, 0, time.Local).Format("2006010215")

		speed := speeds[dateStr]

		fmt.Println(dateStr, speed)
	}

	fmt.Println("")

	for hour := 0; hour < 24; hour++ {
		dateStr := time.Date(2015, 1, 3, hour, 0, 0, 0, time.Local).Format("2006010215")

		_, as := network.FeedForward(mat.NewColVector(4, []float64{(2015.0 - 2010.0) / 5.0, (1.0 - 1.0) / 11.0, (3.0 - 1.0) / 30.0, float64(hour) / 23.0}))
		speed := as[len(as) - 1].Data[0] * 150.0

		fmt.Println(dateStr, speed)
	}

	//for _, td := range tds[:20] {
	//	input := td.In
	//	output := td.Out
	//	_, as := network.FeedForward(input)
	//
	//	log.Println()
	//	log.Println(input.Data[0] * 5.0 + 2010.0, input.Data[1] * 11.0 + 1, input.Data[2] * 30.0 + 1, input.Data[3] * 23.0, ", speed: ", output.Data[0] * 150.0)
	//	log.Println(as[len(as) - 1].Data[0] * 150.0)
	//}

	nn.SaveNetwork("./net_VehicleSpeed.json", network)
}

func plotCosts(lines int, amp float64, costs []float64) {
	for i := 0; i < lines; i++ {
		index := int(float64(len(costs)) * (float64(i) / float64(lines)))
		cost := costs[index]
		buffer := bytes.NewBufferString("")
		buffer.WriteString(fmt.Sprintf("%04v ", index))
		for j := 0; j < int(cost*amp); j++ {
			buffer.WriteString(fmt.Sprint("#"))
		}
		buffer.WriteString(fmt.Sprintf("%.2v\n", cost))
		log.Print(buffer.String())
	}
}
