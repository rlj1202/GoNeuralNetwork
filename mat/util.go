package mat

import "math"

func Argmax (a []float64) int {
	index := 0
	max := 0.0

	for i, value := range a {
		if value > max {
			index = i
			max = value
		}
	}

	return index
}

func Softmax (a []float64) []float64 {
	r := make([]float64, len(a))

	sum := 0.0
	for _, value := range a {
		sum += math.Exp(value)
	}

	for i, value := range a {
		r[i] = math.Exp(value) / sum
	}

	return r
}

func Permutations(input []int) [][]int {
	arr := make([]int, len(input))
	for i, value := range input {
		arr[i] = value
	}

	var helper func([]int, int)
	res := [][]int{}

	helper = func(arr []int, n int){
		if n == 1{
			tmp := make([]int, len(arr))
			copy(tmp, arr)
			res = append(res, tmp)
		} else {
			for i := 0; i < n; i++{
				helper(arr, n - 1)
				if n % 2 == 1{
					tmp := arr[i]
					arr[i] = arr[n - 1]
					arr[n - 1] = tmp
				} else {
					tmp := arr[0]
					arr[0] = arr[n - 1]
					arr[n - 1] = tmp
				}
			}
		}
	}
	helper(arr, len(arr))

	return res
}

func Remap(value, fromMin, fromMax, toMin, toMax float64) float64 {
	fromLen := fromMax - fromMin
	toLen := toMax - toMin

	return ((value - fromMin) / fromLen) * toLen + toMin
}
