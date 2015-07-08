package nn

import (
	"fmt"
	"log"
	"math"
	//	"math/rand"
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
	n.AddHidden(5, true, LogisticActivation)
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

func train() (*ANN, float64, [][]float64) {
	n := NewANN(1, 1, LogisticActivation)
	n.AddHidden(2, true, LogisticActivation)
	n.AddHidden(2, true, LogisticActivation)
	n.Wire()

	feats := make([][]float64, 50)
	targs := make([][]float64, 50)
	for i := 0; i < 50; i++ {
		feats[i] = []float64{float64(i) / 50 * math.Pi}
		targs[i] = []float64{math.Sin(feats[i][0])}
	}

	_, terr := n.Train(feats, targs)
	return n, terr, feats
}

func TestTrain(t *testing.T) {
	numTrys := 1
	n, err, feats := train()
	for math.Abs(err) > 0.1 {
		n, err, feats = train()
		numTrys += 1
	}
	fmt.Printf("Number of trys: %d\n", numTrys)
	fmt.Printf("Error: %f\n", err)
	fmt.Println("Trained with:")
	for i := 0; i < len(feats); i++ {
		fmt.Printf("%f,%f\n", feats[i][0], math.Sin(feats[i][0]))
	}
	fmt.Println("\nRandom test:")
	for i := 0; i < 50; i++ {
		x := float64(i) / 50 * math.Pi
		r := n.Predict([]float64{x})
		fmt.Printf("%f,%f\n", x, r[0])
	}
}
