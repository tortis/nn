package nn

import (
	"log"
	"testing"
)

func TestNNCreate(t *testing.T) {
	n := NewANN(10, 1, LogisticActivation)
	n.AddHidden(5, true, LogisticActivation)
	if n.NumConnections() != 61 {
		t.Fatalf("The created network does not have the expected number of connections. %d != %d", 61, n.NumConnections())
	}
	n.Wire()
}

func TestNNRun(t *testing.T) {
	n := NewANN(10, 5, LogisticActivation)
	n.AddHidden(3, true, LogisticActivation)
	n.Wire()
	out := n.Predict([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
	log.Printf("Output vector: %v\n", out)
}

func TestWeightLoadSave(t *testing.T) {
	// Create a new NN
	n := NewANN(10, 1, LogisticActivation)
	n.AddHidden(5, true, LogisticActivation)
	n.Wire()
	firstSavedWeights := n.SaveWeights()
	log.Printf("Num weights saved: %d\n", len(firstSavedWeights))
	n.LoadWeights(firstSavedWeights)
	secondSavedWeights := n.SaveWeights()
	if len(firstSavedWeights) != len(secondSavedWeights) {
		t.Fatalf("Number of weights changed after saving. %d -> %d", len(firstSavedWeights), len(secondSavedWeights))
	}
	for i, _ := range firstSavedWeights {
		if firstSavedWeights[i] != secondSavedWeights[i] {
			t.Fatalf("Weights at index %d do not match.", i)
		}
	}
}

func TestTrain(t *testing.T) {
	n := NewANN(1, 1, LogisticActivation)
	n.AddHidden(3, true, LogisticActivation)
	n.Wire()

	feats := [][]float64{
		{},
		{},
		{},
		{},
		{},
		{},
		{},
		{},
		{},
		{},
		{},
		{},
	}
	targs := [][]float64{
		{},
		{},
		{},
		{},
		{},
		{},
		{},
		{},
		{},
		{},
		{},
		{},
	}

}
