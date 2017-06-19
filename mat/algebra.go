package mat

import (
	"bytes"
	"fmt"
)

type Matrix struct {
	Rows int `json:"rows"`
	Cols int `json:"cols"`
	Data []float64 `json:"data"`
}

// Returns new matrix struct. It automatically resizes length of data if it is needed.
func NewMatrix(rows int, cols int, data []float64, init float64) Matrix {
	n_data := rows * cols

	if len(data) > n_data {
		data = data[:n_data]
	} else if len(data) < n_data {
		tmp := make([]float64, n_data - len(data))

		for i := 0; i < len(tmp); i++ {
			tmp[i] = init
		}

		data = append(data, tmp...)
	}
	//} else if data == nil {
	//	data = make([]float64, n_data)
	//
	//	for i := 0; i < len(data); i++ {
	//		data[i] = init
	//	}
	//}

	return Matrix{rows, cols, data}
}

func NewRowVector(dimension int, data []float64, init float64) Matrix {
	return NewMatrix(1, dimension, data, init)
}

func NewColVector(dimension int, data []float64, init float64) Matrix {
	return NewMatrix(dimension, 1, data, init)
}

func (mat Matrix) String() string {
	buffer := bytes.NewBufferString("")

	for i := 0; i < mat.Cols * 5 + 3; i++ { buffer.WriteString("-") }
	buffer.WriteString("\n")

	for row := 1; row <= mat.Rows; row++ {
		buffer.WriteString("| ")
		for col := 1; col <= mat.Cols; col++ {
			buffer.WriteString(fmt.Sprintf("%#-4.2v ", mat.At(row, col)))
		}
		buffer.WriteString("|\n")
	}
	for i := 0; i < mat.Cols * 5 + 3; i++ { buffer.WriteString("-") }
	buffer.WriteString("\n")

	return buffer.String()
}

func (mat Matrix) At(row int, col int) float64 {
	return mat.Data[(row - 1) * mat.Cols + (col - 1)]
}

func (mat Matrix) Set(row int, col int, value float64) {
	mat.Data[(row - 1) * mat.Cols + (col - 1)] = value
}

func (mat Matrix) Transpose() Matrix {
	r := NewMatrix(mat.Cols, mat.Rows, make([]float64, mat.Rows * mat.Cols), 0)

	for row := 1; row <= mat.Rows; row++ {
		for col := 1; col <= mat.Cols; col++ {
			r.Set(col, row, mat.At(row, col))
		}
	}

	return r
}

// Matrix product.
func (a Matrix) MatProd(b Matrix) Matrix {
	if a.Cols != b.Rows {
		panic("matrix multiplication error: shapes between two matrices are not valid: ")// TODO
	}

	r := NewMatrix(a.Rows, b.Cols, make([]float64, a.Rows * b.Cols), 0)

	for row := 1; row <= r.Rows; row++ {
		for col := 1; col <= r.Cols; col++ {
			sum := 0.0

			for i := 1; i <= a.Cols; i++ {
				sum += a.At(row, i) * b.At(i, col)
			}

			r.Set(row, col, sum)
		}
	}

	return r
}

func (mat Matrix) Apply(f func(float64) float64) Matrix {
	r := NewMatrix(mat.Rows, mat.Cols, nil, 0)

	for row := 1; row <= r.Rows; row++ {
		for col := 1; col <= r.Cols; col++ {
			r.Set(row, col, f(mat.At(row, col)))
		}
	}

	return r
}

func (a Matrix) ApplyWith(b Matrix, f func(float64, float64) float64) Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("panic in ApplyWith: shapes between two matices are not equal: {%d, %d}, {%d, %d}\n", a.Rows, a.Cols, b.Rows, b.Cols))
	}

	r := NewMatrix(a.Rows, a.Cols, nil, 0)

	for row := 1; row <= r.Rows; row++ {
		for col := 1; col <= r.Cols; col++ {
			r.Set(row, col, f(a.At(row, col), b.At(row, col)))
		}
	}

	return r
}

// Element-wise add.
func (a Matrix) Add(b Matrix) Matrix {
	return a.ApplyWith(b, func (a float64, b float64) float64 { return a + b })
}

// Element-wise sub.
func (a Matrix) Sub(b Matrix) Matrix {
	return a.ApplyWith(b, func (a float64, b float64) float64 { return a - b })
}

// Element-wise mul. Hadamard product.
func (a Matrix) Mul(b Matrix) Matrix {
	return a.ApplyWith(b, func (a float64, b float64) float64 { return a * b })
}
