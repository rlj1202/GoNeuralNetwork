package kec

import (
	"os"
	"bufio"
	"io"
	"strings"
	"encoding/csv"
	"strconv"
)

var (
	coneZoneFileName = "./kec/ETC_conezone.txt"

	fileNames = []string{
		"./kec/VDS_conezone_speed_201401.txt",
		"./kec/VDS_conezone_speed_201402.txt",
		"./kec/VDS_conezone_speed_201403.txt",
		"./kec/VDS_conezone_speed_201404.txt",
		"./kec/VDS_conezone_speed_201405.txt",
		"./kec/VDS_conezone_speed_201406.txt",
		"./kec/VDS_conezone_speed_201407.txt",
		"./kec/VDS_conezone_speed_201408.txt",
		"./kec/VDS_conezone_speed_201409.txt",
		"./kec/VDS_conezone_speed_201410.txt",
		"./kec/VDS_conezone_speed_201411.txt",
		"./kec/VDS_conezone_speed_201412.txt",
		"./kec/VDS_conezone_speed_201501.txt",
		"./kec/VDS_conezone_speed_201502.txt",
		"./kec/VDS_conezone_speed_201503.txt",
		"./kec/VDS_conezone_speed_201504.txt",
		"./kec/VDS_conezone_speed_201505.txt",
		"./kec/VDS_conezone_speed_201506.txt",
		"./kec/VDS_conezone_speed_201507.txt",
		"./kec/VDS_conezone_speed_201508.txt",
		"./kec/VDS_conezone_speed_201509.txt",
		"./kec/VDS_conezone_speed_201510.txt",
		"./kec/VDS_conezone_speed_201511.txt",
		"./kec/VDS_conezone_speed_201512.txt",
	}

	coneZoneNames map[string]string
)

func loadConeZoneNames() {
	if coneZoneNames == nil {
		coneZoneNames = make(map[string]string, 0)
	}

	var (
		err error

		record []string
	)

	file, err := os.Open(coneZoneFileName)
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

func GetConeZoneName(id string) string {
	if coneZoneNames == nil {
		loadConeZoneNames()
	}

	return coneZoneNames[id]
}

func GetVehicleSpeeds(coneZoneID string) map[string]float64 {
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
