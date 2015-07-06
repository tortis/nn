package nn

import (
	"math"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

type ActivationFunction func(float64) float64

type Connection struct {
	from   *Node
	to     *Node
	weight float64
}

func NewRandConnection(from, to *Node) *Connection {
	return &Connection{
		from:   from,
		to:     to,
		weight: rand.Float64() / 5.0,
	}
}

type Node struct {
	out      []*Connection
	in       []*Connection
	val      float64
	activate ActivationFunction
}

type Layer []*Node

func NewLayer(size int, fn ActivationFunction) Layer {
	ns := make([]*Node, size)
	// Set the activation functions
	for i, _ := range ns {
		ns[i] = &Node{
			activate: fn,
		}
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
		input:  NewLayer(inputSize, activate),
		output: NewLayer(outputSize, activate),
	}
}

func (this *ANN) addHidden(size int, fn ActivationFunction) {
	this.hidden = append(this.hidden, NewLayer(size, fn))
}

func (this *ANN) wire() {
	// Connect input to output if there are no hidden layers
	if len(this.hidden) == 0 {
		for _, inputNode := range this.input {
			// Connect input directly to output
			for _, outputNode := range this.output {
				c := NewRandConnection(inputNode, outputNode)
				inputNode.out = append(inputNode.out, c)
				outputNode.in = append(outputNode.in, c)
			}
		}
		return
	}

	// Connect input to the first hidden layer
	for _, inputNode := range this.input {
		for _, hiddenNode := range this.hidden[0] {
			c := NewRandConnection(inputNode, hiddenNode)
			inputNode.out = append(inputNode.out, c)
			hiddenNode.in = append(hiddenNode.in, c)
		}
	}

	// Create connections from hidden to hidden
	if len(this.hidden) >= 2 {
		for i := 0; i < len(this.hidden)-1; i++ {
			for _, hiddenNode := range this.hidden[i] {
				for _, nextHiddenNode := range this.hidden[i+1] {
					c := NewRandConnection(hiddenNode, nextHiddenNode)
					hiddenNode.out = append(hiddenNode.out, c)
					nextHiddenNode.in = append(nextHiddenNode.in, c)
				}
			}
		}
	}

	// Create connection from hidden to output
	for _, hiddenNode := range this.hidden[len(this.hidden)-1] {
		for _, outputNode := range this.output {
			c := NewRandConnection(hiddenNode, outputNode)
			hiddenNode.out = append(hiddenNode.out, c)
			outputNode.in = append(outputNode.in, c)
		}
	}
}

func (this *ANN) run() {
	// Run input layer
	for _, inputNode := range this.input {
		a := inputNode.activate(inputNode.val)
		for _, con := range inputNode.out {
			con.to.val += con.weight * a
		}
	}

	// Run hidden layers
	for i, _ := range this.hidden {
		for _, hiddenNode := range this.hidden[i] {
			a := hiddenNode.activate(hiddenNode.val)
			for _, con := range hiddenNode.out {
				con.to.val += con.weight * a
			}
		}
	}
}

func (this *ANN) reset() {
	for i, _ := range this.hidden {
		for _, hiddenNode := range this.hidden[i] {
			hiddenNode.val = 0.0
		}
	}

	for _, outputNode := range this.output {
		outputNode.val = 0.0
	}
}

func StepActivation(in float64) float64 {
	if in > 0 {
		return 1.0
	} else {
		return 0
	}
}

func LogisticActivation(in float64) float64 {
	return 1.0 / (1.0 + math.Exp(-10*in))
}
