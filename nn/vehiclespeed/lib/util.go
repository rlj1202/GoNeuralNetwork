package vehiclespeed

import (
	"github.com/rlj1202/GoNeuralNetwork/nn"
	"os"
	"encoding/csv"
	"io"
	"strconv"
	"github.com/rlj1202/GoNeuralNetwork/mat"
	"github.com/rlj1202/GoNeuralNetwork/kec"
)

func SaveVehicleSpeeds(dataFrom, id, fileName string) {
	file, err := os.OpenFile(fileName, os.O_WRONLY | os.O_CREATE, 0666)
	defer file.Close()

	if err != nil {
		panic(err)
	}

	writer := csv.NewWriter(file)
	writer.Write([]string{"year", "month", "day", "hour", "speed"})

	speeds := kec.GetVehicleSpeeds(dataFrom, id)

	for _, data := range speeds {
		writer.Write([]string{
			strconv.FormatInt(int64(data.Year), 10),
			strconv.FormatInt(int64(data.Month), 10),
			strconv.FormatInt(int64(data.Day), 10),
			strconv.FormatInt(int64(data.Hour), 10),
			strconv.FormatFloat(data.Speed, 'f', -1, 64),
		})
	}

	writer.Flush()
	file.Sync()
}

func LoadVehicleSpeeds(fileName string) []nn.TrainingData {
	tds := make([]nn.TrainingData, 0)

	file, err := os.OpenFile(fileName, os.O_RDONLY, 0666)

	if err != nil {
		panic(err)
	}

	reader := csv.NewReader(file)
	record, err := reader.Read()

	for {
		if record, err = reader.Read(); err != nil {
			if err == io.EOF {
				break
			} else {
				panic(err)
			}
		}

		year, _ := strconv.ParseInt(record[0], 10, 64)
		month, _ := strconv.ParseInt(record[1], 10, 64)
		day, _ := strconv.ParseInt(record[2], 10, 64)
		hour, _ := strconv.ParseInt(record[3], 10, 64)
		speed, _ := strconv.ParseFloat(record[4], 64)

		input := mat.NewColVector(4, []float64{float64(year - 2010) / 5.0, float64(month - 1) / 11.0, float64(day - 1) / 30.0, float64(hour) / 23.0}, 0)
		output := mat.NewColVector(1, []float64{speed / 150.0}, 0)
		td := nn.TrainingData{In: input, Out: output}
		tds = append(tds, td)
	}

	return tds
}
