package activation

import "math"

func Step(in float64) float64 {
	if in > 0 {
		return 1.0
	} else {
		return 0
	}
}

func Linear(x float64) float64 {
	return 4.0 * x
}

func Logistic(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
