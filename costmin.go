package nn

import "math"

func (this *ANN) Train(feat, targ [][]float64) (int, float64) {
	gradientAscent := func(costfn func([]float64) float64, weights []float64, step float64) int {
		epsilon := 1e-4
		feval := costfn(weights)
		grad := make([]float64, len(weights))
		momentum := make([]float64, len(weights))
		newweights := make([]float64, len(weights))
		numItr := 0

		for {
			preEval := feval
			// Calculate the grad
			for i := 0; i < len(weights); i++ {
				val := weights[i]
				weights[i] = val + epsilon
				right := costfn(weights)
				weights[i] = val - epsilon
				left := costfn(weights)
				weights[i] = val
				grad[i] = (right - left) / (2 * epsilon)
				grad[i] += 0.9 * momentum[i]
				momentum[i] = momentum[i]*0.5 + grad[i]*0.5
			}

			// New weights
			for i := 0; i < len(weights); i++ {
				newweights[i] = weights[i] + step*grad[i]
			}

			feval = costfn(newweights)
			numItr += 1

			if math.Abs(feval) < math.Abs(preEval) {
				if numItr > 500 && math.Abs(math.Abs(feval)-math.Abs(preEval)) < epsilon {
					return numItr
				}
				for i := 0; i < len(weights); i++ {
					weights[i] = newweights[i]
				}
			} else {
				return numItr
			}
		}
		return numItr
	}

	cost := func(weights []float64) float64 {
		// Load the weights
		this.LoadWeights(weights)

		// Comput the cost
		J := 0.0
		for m := 0; m < len(feat); m++ {
			prediction := this.Predict(feat[m])
			sqerr := 0.0
			for k := 0; k < len(prediction); k++ {
				sqerr += math.Pow(targ[m][k]-prediction[k], 2)
			}
			J += sqerr
		}
		J /= float64(len(feat))

		// Regularisation
		R := 0.0
		for i := 0; i < len(weights); i++ {
			R += math.Pow(weights[i], 2)
		}
		R *= 0.01 / float64(this.NumConnections())

		return -(J + R)
	}
	weights := this.SaveWeights()
	numItr := gradientAscent(cost, weights, 1)
	terr := cost(weights)
	return numItr, terr
}
