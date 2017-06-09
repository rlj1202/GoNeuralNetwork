package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
)

const (
	U_BYTE = 0x08 // byte, uint8
	BYTE   = 0x09 // int8
	SHORT  = 0x0b // int16
	INT    = 0x0c // int32
	FLOAT  = 0x0d // float32
	DOUBLE = 0x0e // float64
)

type IDX struct {
	dataType        int
	dimension       int
	sizeInDimension []uint32
	data            []byte
}

func (idx IDX) GetDataType() int {
	return idx.dataType
}

func (idx IDX) GetDimension() int {
	return idx.dimension
}

func (idx IDX) GetSizeInDimension(dimension int) uint32 {
	return idx.sizeInDimension[dimension]
}

func (idx IDX) Get(indices ...int) []byte {
	if len(indices) > idx.dimension {
		panic(fmt.Sprintln("indices length out of dimension"))
	}

	result := idx.data

	for dimension, index := range indices {
		if uint32(index) >= idx.sizeInDimension[dimension] {
			panic(fmt.Sprintln("index out of size in dimension"))
		}

		indexStride := uint32(1)
		for _, size := range idx.sizeInDimension[dimension+1:] {
			indexStride *= size
		}

		//fmt.Printf("dimension %d: stride %d: index %d\n", dimension, indexStride, index)
		//fmt.Printf("data length: %d\n", len(idx.data))

		offset := uint32(index) * indexStride
		result = result[offset : offset+indexStride]
	}

	return result
}

func ReadIDX(file *os.File) (idx IDX) {
	reader := bufio.NewReader(file)

	magic := make([]byte, 4)
	n, err := reader.Read(magic)

	if err != nil {
	}

	if n < 4 {
	}

	idx = IDX{}

	idx.dataType = int(magic[2])
	idx.dimension = int(magic[3])
	idx.sizeInDimension = make([]uint32, idx.dimension)

	for i := 0; i < idx.dimension; i++ {
		size := make([]byte, 4)
		reader.Read(size)
		idx.sizeInDimension[i] = binary.BigEndian.Uint32(size)
	}

	idx.data = make([]byte, 0)
	buffer := make([]byte, 4096)
	for {
		n, err := reader.Read(buffer)

		if err != nil {
		}

		idx.data = append(idx.data, buffer[:n]...)

		if n == 0 {
			break
		}
	}

	return
}
