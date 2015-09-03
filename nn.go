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

func (this *ANN) Reset() {
	for _, inNode := range this.input {
		inNode.val = 0
	}
	for _, outNode := range this.output {
		outNode.val = 0
	}

	for _, hLayer := range this.hidden {
		for _, hNode := range hLayer {
			hNode.val = 0
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

func (this *ANN) Dream(output []float64, invFn ActivationFunction) []float64 {
	if len(this.hidden) != 1 {
		log.Fatal("Dream only works with 3 layer networks.")
	}
	if len(output) != len(this.output) {
		log.Fatal("The output length given does not match the number of output nodes.")
	}

	this.Reset()

	// First set the output nodes values, passing it through the inverse activation function
	for i, outval := range output {
		this.output[i].val = invFn(outval)
	}

	// Now compute the post activation values of the hidden layers
	for _, outNode := range this.output {
		// First compute the sum of all input weights
		totalWeight := 0.0
		for _, inConn := range outNode.in {
			totalWeight += inConn.weight
		}

		// Now add the contributing portion to each hidden node
		for _, inConn := range outNode.in {
			inConn.from.val += inConn.weight / totalWeight * outNode.val
		}
	}

	// Compute pre-activation for hidden layers

	return make([]float64, 0)
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
