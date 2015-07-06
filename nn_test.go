package nn

import (
	"log"
	"testing"
)

func TestNNCreate(t *testing.T) {
	n := NewANN(10, 1, LogisticActivation)
	n.addHidden(5, LogisticActivation)
	n.wire()
}

func TestNNRun(t *testing.T) {
	n := NewANN(10, 1, LogisticActivation)
	n.addHidden(5, LogisticActivation)
	n.wire()
	for _, inputNode := range n.input {
		inputNode.val = 1.0
	}
	n.run()
	for _, outputNode := range n.output {
		log.Printf("Output Node: %v\n", outputNode)
	}
}
