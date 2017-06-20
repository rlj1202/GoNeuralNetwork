package kec

import (
	"testing"
)

func TestConeZone(t *testing.T) {
	loadConeZoneNames("./")

	id := "0010CZS440"

	speeds := GetVehicleSpeeds("./", id)

	for _, data := range speeds {
		t.Log(data.Year, data.Month, data.Day, data.Hour, " | ", data.Speed, "km/h")
	}

	t.Log(GetConeZoneName("./", id))
}
