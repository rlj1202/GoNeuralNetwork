package kec

import (
	"time"
	"strconv"
	"testing"
)

func TestConeZone(t *testing.T) {
	loadConeZoneNames()

	id := "0010CZS440"

	speeds := GetVehicleSpeeds(id)

	for dateStr, speed := range speeds {
		date, _ := time.Parse("2006010215", dateStr)

		t.Log(date, strconv.FormatFloat(speed, 'f', 6, 64), "km/h")
	}

	t.Log(GetConeZoneName(id))
}
