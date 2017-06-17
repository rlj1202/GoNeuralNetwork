package main

import (
	"fmt"
	"time"
	"strconv"
)

func main() {
	loadConeZoneNames()

	id := "0010CZS440"

	speeds := load(id)

	for dateStr, speed := range speeds {
		date, _ := time.Parse("2006010203", dateStr)

		fmt.Println(date, strconv.FormatFloat(speed, 'f', 6, 64) + " km/h")
	}

	fmt.Println(getConeZoneName(id))
}
