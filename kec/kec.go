package main

import (
	"os"
	"bufio"
	"io"
	"strings"
	"encoding/csv"
	"strconv"
)

var (
	conezoneFileName = "./ETC_conezone.txt"

	fileNames = []string{
		"./VDS_conezone_speed_201401.txt",
		"./VDS_conezone_speed_201402.txt",
		"./VDS_conezone_speed_201403.txt",
		"./VDS_conezone_speed_201404.txt",
	}

	coneZoneNames = map[string]string{}
)

func loadConeZoneNames() {
	var (
		err error

		record []string
	)

	file, err := os.Open(conezoneFileName)
	reader := csv.NewReader(file)

	for {
		if record, err = reader.Read(); err != nil {
			if err == io.EOF {
				break
			} else {
				panic(err)
			}
		}

		id := record[0]
		name := record[9]

		coneZoneNames[id] = name
	}
}

func getConeZoneName(id string) string {
	return coneZoneNames[id]
}

func load(coneZoneID string) map[string]float64 {
	result := map[string]float64{}

	var (
		err error

		file *os.File
		reader *bufio.Reader

		part []byte
	)

	for _, fileName := range fileNames {
		file, err = os.Open(fileName)

		if err != nil {
			panic(err)
		}

		reader = bufio.NewReader(file)

		for {
			if part, _, err = reader.ReadLine(); err != nil {
				if err == io.EOF {
					break
				} else {
					panic(err)
				}
			}

			rawData := strings.Split(string(part), "|")

			date := rawData[0]
			hour := rawData[1]
			coneZone := rawData[2]

			roadType0, _ := strconv.ParseInt(rawData[3], 10, 0)
			roadType := int(roadType0)

			vehicleSpeed, _ := strconv.ParseFloat(rawData[4], 64)

			if coneZoneID == coneZone && roadType == 1 {
				result[date + hour] = vehicleSpeed
			}
		}
	}

	return result
}
