package main

import (
	"os"
	"log"
	"time"
	"github.com/rlj1202/GoNeuralNetwork/nn"
)

func main() {
	logFileName := "log_"
	logFileName = string(append([]byte(logFileName), []byte(time.Now().Format("2006-01-02_03-04"))...))
	logFileName = string(append([]byte(logFileName), []byte(".txt")...))
	logFile, err := os.OpenFile(logFileName, os.O_CREATE | os.O_WRONLY | os.O_APPEND, 0666)
	if err != nil {
		panic(err)
	}
	defer logFile.Close()

	log.SetOutput(logFile)

	costs := nn.ConcatCostsResult(
		"./HandWriting_2017-06-10_03-36.csv",
		"./HandWriting_2017-06-10_04-19.csv",
		"./HandWriting_2017-06-10_09-50.csv",
		"./HandWriting_2017-06-10_10-42.csv",
		"./HandWriting_2017-06-11_01-25.csv",
		"./HandWriting_2017-06-14_01-43.csv")
	nn.WriteCostsResult("./HandWriting_total.csv", costs)
}
