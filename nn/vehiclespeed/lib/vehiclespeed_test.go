package vehiclespeed_test

import (
	"testing"
	"github.com/rlj1202/GoNeuralNetwork/nn/vehiclespeed/lib"
	"os"
	//"path/filepath"
)

func Test(t *testing.T) {
	t.Log(os.Getwd())

	id := "0010CZS440"

	//path, _ := filepath.Abs("../../../kec")
	//vehiclespeed.SaveVehicleSpeeds(path, id, "speeds_" + id + ".csv")

	tds := vehiclespeed.LoadVehicleSpeeds(id, "speeds_" + id + ".csv")

	for _, td := range tds {
		year := td.In.Data[0] * 5.0 + 2010
		month := td.In.Data[1] * 11.0 + 1
		day := td.In.Data[2] * 30.0 + 1
		hour := td.In.Data[3] * 23.0

		speed := td.Out.Data[0] * 150.0

		t.Log(year, month, day, hour, speed)
	}
}
