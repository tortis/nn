package nn

import (
	"log"
	"math"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

type ActivationFunction func(float64) float64

type Connection struct {
	from     *Node
	to       *Node
	weight   float64
	momentum float64
}

// Initialize a connection weight using the method described here:
// http://ai-maker.com/artificial-neural-networks-to-the-point/
// "The parameter initialisation process is based on a uniform distribution
// between two small numbers that take into account the amount of input and
// output units of the adjacent layers."
func NewRandConnection(out, in int, from, to *Node) *Connection {
	max := 2.44948974 / (math.Sqrt(float64(in) + float64(out)))
	return &Connection{
		from:   from,
		to:     to,
		weight: rand.Float64()*max*2 - max, // In [-max, max]
	}
}

type Node struct {
	out         []*Connection
	in          []*Connection
	val         float64
	activate    ActivationFunction
	errorSignal float64
	isBias      bool
}

type Layer []*Node

func newLayer(size int, withBias bool, fn ActivationFunction) Layer {
	ns := make([]*Node, size, size+1)
	// Set the activation functions
	for i := 0; i < size; i++ {
		ns[i] = &Node{
			out:      make([]*Connection, 0),
			in:       make([]*Connection, 0),
			activate: fn,
		}
	}
	if withBias {
		ns = append(ns, &Node{
			out:    make([]*Connection, 0),
			in:     make([]*Connection, 0),
			val:    1.0,
			isBias: true,
		})
	}
	return ns
}

type ANN struct {
	input  Layer
	hidden []Layer
	output Layer
}

func NewANN(inputSize, outputSize int, activate ActivationFunction) *ANN {
	return &ANN{
		input:  newLayer(inputSize, true, activate),
		output: newLayer(outputSize, false, activate),
	}
}

func (this *ANN) AddHidden(size int, withBias bool, fn ActivationFunction) {
	this.hidden = append(this.hidden, newLayer(size, withBias, fn))
}

func (this *ANN) Wire() {
	// Connect input to output if there are no hidden layers
	if len(this.hidden) == 0 {
		for i := 0; i < len(this.input); i++ {
			inputNode := this.input[i]
			// Connect input directly to output
			for j := 0; j < len(this.output); j++ {
				outputNode := this.output[j]
				c := NewRandConnection(len(this.input), len(this.output), inputNode, outputNode)
				inputNode.out = append(inputNode.out, c)
				outputNode.in = append(outputNode.in, c)
			}
		}
		return
	}

	// Connect input to the first hidden layer
	for i := 0; i < len(this.input); i++ {
		inputNode := this.input[i]
		for j := 0; j < len(this.hidden[0]); j++ {
			hiddenNode := this.hidden[0][j]
			if hiddenNode.isBias {
				continue
			}
			c := NewRandConnection(len(this.input), len(this.hidden[0]), inputNode, hiddenNode)
			inputNode.out = append(inputNode.out, c)
			hiddenNode.in = append(hiddenNode.in, c)
		}
	}

	// Create connections from hidden to hidden
	if len(this.hidden) >= 2 {
		for i := 0; i < len(this.hidden)-1; i++ {
			for j := 0; j < len(this.hidden[i]); j++ {
				hiddenNode := this.hidden[i][j]
				for k := 0; k < len(this.hidden[i+1]); k++ {
					nextHiddenNode := this.hidden[i+1][k]
					if nextHiddenNode.isBias {
						continue
					}
					c := NewRandConnection(len(this.hidden[i]), len(this.hidden[i+1]), hiddenNode, nextHiddenNode)
					hiddenNode.out = append(hiddenNode.out, c)
					nextHiddenNode.in = append(nextHiddenNode.in, c)
				}
			}
		}
	}

	// Create connection from hidden to output
	for i := 0; i < len(this.hidden[len(this.hidden)-1]); i++ {
		hiddenNode := this.hidden[len(this.hidden)-1][i]
		for j := 0; j < len(this.output); j++ {
			outputNode := this.output[j]
			c := NewRandConnection(len(this.hidden[len(this.hidden)-1]), len(this.output), hiddenNode, outputNode)
			hiddenNode.out = append(hiddenNode.out, c)
			outputNode.in = append(outputNode.in, c)
		}
	}
}

func (this *ANN) Predict(input []float64) []float64 {
	if len(input) > len(this.input) {
		log.Fatal("The input vector is too long.")
	} else if len(input) < len(this.input)-1 {
		log.Fatal("The input vector is too short.")
	}
	// Set the input values
	for i, inval := range input {
		if this.input[i].isBias {
			log.Println("Warning: Assigning input value to the input bias node.")
		}
		this.input[i].val = inval
	}

	// Run the network
	// For each hidden node, compute the sum of all its inputs,
	// then run that sum through its activation function.
	for i, _ := range this.hidden {
		for j := 0; j < len(this.hidden[i]); j++ {
			hiddenNode := this.hidden[i][j]
			if hiddenNode.isBias {
				continue
			}
			// Compute the value of the node
			hiddenNode.val = 0
			for k := 0; k < len(hiddenNode.in); k++ {
				hiddenNode.val += hiddenNode.in[k].from.val * hiddenNode.in[k].weight
			}
			// Pass the sumed inputs through the activation function
			hiddenNode.val = hiddenNode.activate(hiddenNode.val)
		}
	}

	// Compute the output vector
	output := make([]float64, len(this.output))
	for i := 0; i < len(this.output); i++ {
		outputNode := this.output[i]
		outputNode.val = 0
		for j := 0; j < len(outputNode.in); j++ {
			outputNode.val += outputNode.in[j].from.val * outputNode.in[j].weight
		}
		outputNode.val = outputNode.activate(outputNode.val)
		output[i] = outputNode.val
	}
	return output
}

func (this *ANN) Backprop(feat, targ [][]float64, maxItr int, minErr, lr, mf float64) float64 {
	errfn := func(weights []float64) float64 {
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
		J /= 2.0

		// Regularisation
		R := 0.0
		for i := 0; i < len(weights); i++ {
			R += math.Pow(weights[i], 2)
		}
		R *= 0.01 / float64(len(weights))

		return J + R
	}

	runPattern := func(exin, exout []float64) {
		// Compute the prediction
		prediction := this.Predict(exin)

		// Following the steps from: https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
		// 1. Calculate errors of output nodes
		for i := 0; i < len(this.output); i++ {
			delt := exout[i] - prediction[i]
			signal := delt * prediction[i] * (1 - prediction[i])
			this.output[i].errorSignal = signal
		}

		// 2. Change output layer weights
		for i := 0; i < len(this.output); i++ {
			for j := 0; j < len(this.output[i].in); j++ {
				con := this.output[i].in[j]
				wd := lr*this.output[i].errorSignal*con.from.val + mf*con.momentum
				con.weight += wd
				con.momentum = wd
			}
		}

		for i := 0; i < len(this.hidden); i++ {
			layer := this.hidden[i]
			// 3. Calculate (back-propagate) hidden layer errors
			for j := 0; j < len(layer); j++ {
				node := layer[j]
				delt := 0.0
				for k := 0; k < len(node.out); k++ {
					delt += node.out[k].to.errorSignal * node.out[k].weight
				}
				node.errorSignal = node.val * (1.0 - node.val) * delt
			}

			// 4. Change hidden layer weights
			for j := 0; j < len(layer); j++ {
				node := layer[j]
				for k := 0; k < len(node.in); k++ {
					con := node.in[k]
					wd := lr*node.errorSignal*con.from.val + mf*con.momentum
					con.weight += wd
					con.momentum = wd
				}
			}
		}
	}

	log.Printf("Initial error: %f\n", errfn(this.SaveWeights()))

	for numItr := 0; numItr < maxItr; numItr++ {
		for i := 0; i < len(feat); i++ {
			runPattern(feat[i], targ[i])
		}
		numItr += 1
		if errfn(this.SaveWeights()) < minErr {
			log.Printf("Meet training goal after %d iterations.\n", numItr)
			return errfn(this.SaveWeights())
		}
		//log.Printf("error: %f\n", errfn(this.SaveWeights()))
	}
	log.Printf("Gaveup after %d iterations.\n", 20000)
	return errfn(this.SaveWeights())
}

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

func (this *ANN) LoadWeights(weights []float64) {
	if len(weights) != this.NumConnections() {
		log.Printf("LoadWeights: Given weights vector does not match number of weights in the network. Given %d expected %d\n", len(weights), this.NumConnections())
	}
	wi := 0
	for i := 0; i < len(this.hidden); i++ {
		for j := 0; j < len(this.hidden[i]); j++ {
			hiddenNode := this.hidden[i][j]
			for k := 0; k < len(hiddenNode.in); k++ {
				hiddenNode.in[k].weight = weights[wi]
				wi += 1
			}
		}
	}

	for i := 0; i < len(this.output); i++ {
		outNode := this.output[i]
		for j := 0; j < len(outNode.in); j++ {
			outNode.in[j].weight = weights[wi]
			wi += 1
		}
	}
}

func (this *ANN) SaveWeights() []float64 {
	// Count the expected number of weights for efficency.
	weights := make([]float64, 0, this.NumConnections())
	for i := 0; i < len(this.hidden); i++ {
		for j := 0; j < len(this.hidden[i]); j++ {
			hiddenNode := this.hidden[i][j]
			for k := 0; k < len(hiddenNode.in); k++ {
				weights = append(weights, hiddenNode.in[k].weight)
			}
		}
	}

	for i := 0; i < len(this.output); i++ {
		outNode := this.output[i]
		for j := 0; j < len(outNode.in); j++ {
			weights = append(weights, outNode.in[j].weight)
		}
	}
	return weights
}

func (this *ANN) NumConnections() int {
	num := len(this.input) * (len(this.hidden[0]) - 1)
	for i := 0; i < len(this.hidden); i++ {
		if i == len(this.hidden)-1 {
			num += len(this.hidden[i]) * len(this.output)
		} else {
			num += len(this.hidden[i]) * (len(this.hidden[i+1]) - 1)
		}
	}
	return num
}

func StepActivation(in float64) float64 {
	if in > 0 {
		return 1.0
	} else {
		return 0
	}
}

func LinearActivation(x float64) float64 {
	return 4.0 * x
}

func LogisticActivation(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
