package main

import (
	"github.com/rlj1202/GoNeuralNetwork/nn"
	"github.com/rlj1202/GoNeuralNetwork/kec"
	"time"
	"github.com/rlj1202/GoNeuralNetwork/mat"
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

		in := mat.NewColVector(4, []float64{year, month, day, hour}, 0)
		out := mat.NewColVector(1, []float64{speed / 150.0}, 0)
		td := nn.TrainingData{In: in, Out: out}
		tds = append(tds, td)
	}

	network, _ := nn.LoadNetwork("./net_VehicleSpeed.json")
	if network == nil {
		network = new(nn.Network)
		*network = nn.NewNetwork([]int{4, 30, 30, 30, 1}, nil, nil)
	}
	costs := network.StochasticGradientDescent(tds, 100, 50, 0.01, tds, nn.CE)
	nn.PlotCosts(30, 200.0, costs)

	//for hour := 0; hour < 24; hour++ {
	//	dateStr := time.Date(2015, 1, 3, hour, 0, 0, 0, time.Local).Format("2006010215")
	//
	//	speed := speeds[dateStr]
	//
	//	fmt.Println(dateStr, speed)
	//}
	//
	//fmt.Println("")
	//
	//for hour := 0; hour < 24; hour++ {
	//	dateStr := time.Date(2015, 1, 3, hour, 0, 0, 0, time.Local).Format("2006010215")
	//
	//	_, as := network.FeedForward(mat.NewColVector(4, []float64{(2015.0 - 2010.0) / 5.0, (1.0 - 1.0) / 11.0, (3.0 - 1.0) / 30.0, float64(hour) / 23.0}))
	//	speed := as[len(as) - 1].Data[0] * 150.0
	//
	//	fmt.Println(dateStr, speed)
	//}

	for _, td := range tds[:20] {
		input := td.In
		output := td.Out
		_, as := network.FeedForward(input)

		log.Println()
		log.Println(input.Data[0] * 5.0 + 2010.0, input.Data[1] * 11.0 + 1, input.Data[2] * 30.0 + 1, input.Data[3] * 23.0, ", speed: ", output.Data[0] * 150.0)
		log.Println(as[len(as) - 1].Data[0] * 150.0)
	}

	nn.SaveNetwork("./net_VehicleSpeed.json", network)
}
