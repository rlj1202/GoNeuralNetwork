package main

import (
	"github.com/rlj1202/GoNeuralNetwork/nn"
	"log"
	"flag"
	"time"
	"github.com/rlj1202/GoNeuralNetwork/mat"
	"sort"
	"github.com/rlj1202/GoNeuralNetwork/kec"
	"os"
	"encoding/csv"
	"strconv"
	"io"
)

func main() {
	trainingFlag := flag.Bool("training", false, "Training flag.")
	genFlag := flag.Bool("gen", false, "Generate flag.")
	idFlag := flag.String("id", "0010CZS440", "ConeZone id.")
	testFlag := flag.Bool("test", false, "Test flag.")
	flag.Parse()

	// ConeZone ID
	id := *idFlag

	start := time.Date(2014, 1, 1, 0, 0, 0, 0, time.Local)
	end := time.Date(2016, 1, 1, 0, 0, 0, 0, time.Local)

	oriData, err := LoadVehicleSpeeds("./nn/vehiclespeed/ori_" + id + ".csv")
	if err != nil {
		oriData = kec.GetVehicleSpeeds("./kec", id, start, end)
		err = SaveVehicleSpeeds(oriData, "./nn/vehiclespeed/ori_" + id + ".csv")

		if err != nil {
			panic(err)
		}
	}

	tds := ConvertDataToTrainingData(oriData)

	lookup := 24
	networkFileName := "./net_VehicleSpeed_3.json"
	costResultFileName := "./cost_VehicleSpeed_3.csv"
	network, _ := nn.LoadNetwork(networkFileName)
	if network == nil {
		network = new(nn.Network)
		*network = nn.NewNetwork([]int{5 + lookup, 30, 30, 30, 30, 1}, nil, nil)
	}

	if *trainingFlag {
		costs := make([]float64, 0)
		if _, err := os.Stat(costResultFileName); err == nil {
			costs = nn.ReadCostsResult(costResultFileName)
		}

		costs = append(costs, network.StochasticGradientDescent(tds, 100, 50, 0.1, tds, nn.CE)...)
		nn.PlotCosts(30, 200.0, costs)

		nn.WriteCostsResult(costResultFileName, costs)
	}

	if *genFlag {
		log.Println("Original Data:")
		for t := start; t.Before(end); t = t.Add(time.Hour) {
			speed := SpeedPerHour(oriData, t)

			log.Println(t, " | ", speed, " km/h")
		}

		log.Println("Generated Data:")
		genData := make([]kec.Data, 0)
		for _, td := range tds {
			_, as := network.FeedForward(td.In)

			t, speed := Generalize(td.In.Data, as[len(as) - 1].Data[0])

			log.Println(t, " | ", speed, " km/h")

			genData = append(genData, kec.Data{ConeZone: id, Time: t, Speed: speed})
		}
		SaveVehicleSpeeds(genData, "./nn/vehiclespeed/gen_" + id + ".csv")

		//

		log.Println("Average Original Data:")
		avrOriData := make([]kec.Data, 0)
		for t := start; t.Before(end); t = t.AddDate(0, 0, 1) {
			average := SpeedPerDay(oriData, t)
			log.Println("Average: ", t, average)

			avrOriData = append(avrOriData, kec.Data{ConeZone: id, Time: t, Speed: average})
		}
		SaveVehicleSpeeds(avrOriData, "./nn/vehiclespeed/avr_ori_" + id + ".csv")

		log.Println("Average Generated Data:")
		avrGenData := make([]kec.Data, 0)
		for t := start; t.Before(end); t = t.AddDate(0, 0, 1) {
			average := SpeedPerDay(genData, t)
			log.Println("Average: ", t, average)

			avrGenData = append(avrGenData, kec.Data{ConeZone: id, Time: t, Speed: average})
		}
		SaveVehicleSpeeds(avrGenData, "./nn/vehiclespeed/avr_gen_" + id + ".csv")

		log.Println("Average Original Month Data:")
		avrOriMonthData := make([]kec.Data, 0)
		for t := start; t.Before(end); t = t.AddDate(0, 1, 0) {
			average := SpeedPerMonth(oriData, t)
			log.Println("Average: ", t, average)

			avrOriMonthData = append(avrOriMonthData, kec.Data{ConeZone: id, Time: t, Speed: average})
		}
		SaveVehicleSpeeds(avrOriMonthData, "./nn/vehiclespeed/avr_ori_month_" + id + ".csv")

		log.Println("Average Generated Month Data:")
		avrGenMonthData := make([]kec.Data, 0)
		for t := start; t.Before(end); t = t.AddDate(0, 1, 0) {
			average := SpeedPerMonth(genData, t)
			log.Println("Average: ", t, average)

			avrGenMonthData = append(avrGenMonthData, kec.Data{ConeZone: id, Time: t, Speed: average})
		}
		SaveVehicleSpeeds(avrGenMonthData, "./nn/vehiclespeed/avr_gen_month_" + id + ".csv")

		//

		preOriData, err := LoadVehicleSpeeds("./nn/vehiclespeed/pre_ori_" + id + ".csv")
		if err != nil {
			preOriData = kec.GetVehicleSpeeds("./kec", id, time.Date(2017, 4, 1, 0, 0, 0, 0, time.Local), time.Date(2017, 6, 1, 0, 0, 0, 0, time.Local))
			err = SaveVehicleSpeeds(preOriData, "./nn/vehiclespeed/pre_ori_" + id + ".csv")

			if err != nil {
				panic(err)
			}
		}

		log.Println("Predict")
		preGenData := make([]kec.Data, 0)
		v := make([]float64, 0)
		b := time.Date(2017, 5, 1, 0, 0, 0, 0, time.Local)
		for t := b.Add(time.Hour * time.Duration(-lookup)); t.Before(b); t = t.Add(time.Hour) {
			speed := SpeedPerHour(preOriData, t)
			v = append(v, speed / 150.0)
		}
		for t := b; t.Before(b.AddDate(0, 1, 0)); t = t.Add(time.Hour) {
			_, as := network.FeedForward(mat.NewColVector(5 + lookup, append(
				[]float64{
					mat.Remap(float64(t.Year() -3), 2010, 2020, 0, 1),
					mat.Remap(float64(t.Month()), 1, 12, 0, 1),
					mat.Remap(float64(t.Day()), 1, 31, 0, 1),
					mat.Remap(float64(t.Weekday()), 0, 6, 0, 1),
					mat.Remap(float64(t.Hour()), 0, 23, 0, 1),
				},
				v...,
			), 0))
			speed := as[len(as) - 1].Data[0]
			v = append(v[1:], speed)

			preGenData = append(preGenData, kec.Data{ConeZone: id, Time: t, Speed: speed * 150.0})
		}
		SaveVehicleSpeeds(preGenData, "./nn/vehiclespeed/pre_gen_" + id + ".csv")
	}

	if *testFlag {

	}

	nn.SaveNetwork(networkFileName, network)
}

func SaveVehicleSpeeds(data []kec.Data, fileName string) error {
	file, err := os.OpenFile(fileName, os.O_WRONLY | os.O_CREATE, 0666)
	defer file.Close()

	if err != nil {
		return err
	}

	writer := csv.NewWriter(file)
	writer.Write([]string{"conezone", "year", "month", "day", "hour", "speed"})

	for _, data := range data {
		writer.Write([]string{
			data.ConeZone,
			strconv.FormatInt(int64(data.Time.Year()), 10),
			strconv.FormatInt(int64(data.Time.Month()), 10),
			strconv.FormatInt(int64(data.Time.Day()), 10),
			strconv.FormatInt(int64(data.Time.Hour()), 10),
			strconv.FormatFloat(data.Speed, 'f', -1, 64),
		})
	}

	writer.Flush()
	file.Sync()

	return nil
}

func LoadVehicleSpeeds(fileName string) ([]kec.Data, error) {
	results := make([]kec.Data, 0)

	file, err := os.OpenFile(fileName, os.O_RDONLY, 0666)

	if err != nil {
		return nil, err
	}

	reader := csv.NewReader(file)
	record, err := reader.Read()

	for {
		if record, err = reader.Read(); err != nil {
			if err == io.EOF {
				break
			} else {
				return nil, err
			}
		}

		coneZone := record[0]
		year, _ := strconv.ParseInt(record[1], 10, 64)
		month, _ := strconv.ParseInt(record[2], 10, 64)
		day, _ := strconv.ParseInt(record[3], 10, 64)
		hour, _ := strconv.ParseInt(record[4], 10, 64)
		speed, _ := strconv.ParseFloat(record[5], 64)

		date := time.Date(int(year), time.Month(month), int(day), int(hour), 0, 0, 0, time.Local)

		results = append(results, kec.Data{ConeZone: coneZone, Time: date, Speed: speed})
	}

	return results, nil
}

func ConvertDataToTrainingData(dataSet []kec.Data) []nn.TrainingData {
	tds := make([]nn.TrainingData, 0)

	for _, data := range dataSet {
		arr := []float64{
			mat.Remap(float64(data.Time.Year()), 2010, 2020, 0, 1),
			mat.Remap(float64(data.Time.Month()), 1, 12, 0, 1),
			mat.Remap(float64(data.Time.Day()), 1, 31, 0, 1),
			mat.Remap(float64(data.Time.Weekday()), 0, 6, 0, 1),
			mat.Remap(float64(data.Time.Hour()), 0, 23, 0, 1),
		}

		lookup := 24
		for i := 0; i < lookup; i++ {
			t := data.Time.Add(time.Hour * time.Duration(-lookup + i))
			speed := SpeedPerHour(dataSet, t)

			arr = append(arr, mat.Remap(speed, 0, 150, 0, 1))
		}

		in := mat.NewColVector(5 + lookup, arr, 0)
		out := mat.NewColVector(1, []float64{mat.Remap(data.Speed, 0, 150, 0, 1)}, 0)
		td := nn.TrainingData{In: in, Out: out}

		tds = append(tds, td)
	}

	return tds
}

func Generalize(t []float64, s float64) (time.Time, float64) {
	return time.Date(
		int(mat.Remap(t[0], 0, 1, 2010, 2020)),
		time.Month(mat.Remap(t[1], 0, 1, 1, 12)),
		int(mat.Remap(t[2], 0, 1, 1, 31)),
		int(mat.Remap(t[4], 0, 1, 0, 23)),
		0,
		0,
		0,
		time.Local,
	), mat.Remap(s, 0, 1, 0, 150)
}

func SpeedPerHour(data []kec.Data, t time.Time) float64 {
	i := sort.Search(len(data), func (i int) bool { return data[i].Time.After(t) || data[i].Time.Equal(t) })
	d := data[i]

	if i < len(data) && d.Time.Equal(t) {
		return d.Speed
	} else {
		return 0.0
	}
}

func SpeedPerDay(data []kec.Data, t time.Time) float64 {
	sum := 0.0

	for hour := 0; hour < 24; hour++ {
		speed := SpeedPerHour(data, time.Date(t.Year(), t.Month(), t.Day(), hour, 0, 0, 0, time.Local))

		sum += speed
	}

	return sum / 24.0
}

func SpeedPerMonth(data []kec.Data, t time.Time) float64 {
	from := time.Date(t.Year(), t.Month(), 1, 0, 0, 0, 0, time.Local)
	to := time.Date(t.Year(), t.Month() + 1, 1, 0, 0, 0, 0, time.Local)

	sum := 0.0

	days := 0.0
	for i := from; i.Before(to); i = i.AddDate(0, 0, 1) {
		speed := SpeedPerDay(data, i)

		sum += speed
		days++
	}

	return sum / days
}
