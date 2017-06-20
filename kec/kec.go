package kec

import (
	"os"
	"bufio"
	"io"
	"strings"
	"encoding/csv"
	"strconv"
	"path/filepath"
	"time"
	"sort"
)

type Data struct {
	Year int
	Month int
	Day int
	Hour int
	Speed float64
}
type DataByDate []Data

var (
	coneZoneFileName = "ETC_conezone.txt"

	fileNames = []string{
		"VDS_conezone_speed_201401.txt",
		"VDS_conezone_speed_201402.txt",
		"VDS_conezone_speed_201403.txt",
		"VDS_conezone_speed_201404.txt",
		"VDS_conezone_speed_201405.txt",
		"VDS_conezone_speed_201406.txt",
		"VDS_conezone_speed_201407.txt",
		"VDS_conezone_speed_201408.txt",
		"VDS_conezone_speed_201409.txt",
		"VDS_conezone_speed_201410.txt",
		"VDS_conezone_speed_201411.txt",
		"VDS_conezone_speed_201412.txt",
		"VDS_conezone_speed_201501.txt",
		"VDS_conezone_speed_201502.txt",
		"VDS_conezone_speed_201503.txt",
		"VDS_conezone_speed_201504.txt",
		"VDS_conezone_speed_201505.txt",
		"VDS_conezone_speed_201506.txt",
		"VDS_conezone_speed_201507.txt",
		"VDS_conezone_speed_201508.txt",
		"VDS_conezone_speed_201509.txt",
		"VDS_conezone_speed_201510.txt",
		"VDS_conezone_speed_201511.txt",
		"VDS_conezone_speed_201512.txt",
	}

	coneZoneNames map[string]string
)

func loadConeZoneNames(dirPath string) {
	if coneZoneNames == nil {
		coneZoneNames = make(map[string]string, 0)
	}

	var (
		err error

		record []string
	)

	file, err := os.Open(filepath.Clean(dirPath) + string(filepath.Separator) + coneZoneFileName)
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

func GetConeZoneName(dirPath string, id string) string {
	if coneZoneNames == nil {
		loadConeZoneNames(dirPath)
	}

	return coneZoneNames[id]
}

func GetVehicleSpeeds(dirPath, coneZoneID string) []Data {
	result := make([]Data, 0)

	var (
		err error

		file *os.File
		reader *bufio.Reader

		part []byte
	)

	for _, fileName := range fileNames {
		file, err = os.Open(filepath.Clean(dirPath) + string(filepath.Separator) + fileName)

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

			date, _ := time.Parse("20060102", rawData[0])
			year := date.Year()
			month := int(date.Month())
			day := date.Day()
			hour, _ := strconv.ParseInt(rawData[1], 10, 64)
			coneZone := rawData[2]

			roadType0, _ := strconv.ParseInt(rawData[3], 10, 0)
			roadType := int(roadType0)

			speed, _ := strconv.ParseFloat(rawData[4], 64)

			if coneZoneID == coneZone && roadType == 1 {
				result = append(result, Data{year, month, day, int(hour), speed})
			}
		}
	}

	sort.Sort(DataByDate(result))

	return result
}

func (data DataByDate) Len() int {
	return len(data)
}

func (data DataByDate) Less(i, j int) bool {
	a, b := data[i], data[j]

	if a.Year < b.Year {
		return true
	} else if a.Year == b.Year {
		if a.Month < b.Month {
			return true
		} else if a.Month == b.Month {
			if a.Day < b.Day {
				return true
			} else if a.Day == b.Day {
				if a.Hour < b.Hour {
					return true
				}
			}
		}
	}

	return false
}

func (data DataByDate) Swap(i, j int) {
	data[i], data[j] = data[j], data[i]
}
