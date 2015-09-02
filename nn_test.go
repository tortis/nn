package nn

import (
	"fmt"
	"log"
	"math"
	"testing"

	"github.com/tortis/nn/activation"
)

func TestNNCreate(t *testing.T) {
	n := NewANN(10, 1, activation.Logistic)
	n.AddHidden(5, true, activation.Logistic)
	if n.NumConnections() != 61 {
		t.Fatalf("The created network does not have the expected number of connections. %d != %d", 61, n.NumConnections())
	}
	n.Wire()
}

func TestNNRun(t *testing.T) {
	n := NewANN(10, 5, activation.Logistic)
	n.AddHidden(5, true, activation.Logistic)
	n.Wire()
	out := n.Predict([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
	log.Printf("Output vector: %v\n", out)
}

func TestWeightLoadSave(t *testing.T) {
	// Create a new NN
	n := NewANN(10, 1, activation.Logistic)
	n.AddHidden(5, true, activation.Logistic)
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

func train(n *ANN) (float64, [][]float64) {
	feats := make([][]float64, 30)
	targs := make([][]float64, 30)
	for i := 0; i < 30; i++ {
		x := float64(i) / 30 * math.Pi
		//x := rand.Float64() * math.Pi
		feats[i] = []float64{x}
		targs[i] = []float64{math.Sin(feats[i][0])}
	}

	_, terr := n.Train(feats, targs)
	return terr, feats
}

func OffTestTrain(t *testing.T) {
	n := NewANN(1, 1, activation.Logistic)
	n.AddHidden(3, true, activation.Logistic)
	n.Wire()

	besterr, bestfeats := train(n)

	numTrys := 1
	err, feats := train(n)
	for i := 0; i < 10000; i++ {
		if math.Abs(err) < math.Abs(besterr) {
			besterr, bestfeats = err, feats
			log.Printf("Error: %f\n", besterr)
		}
		err, feats = train(n)
		numTrys += 1
	}
	fmt.Printf("Number of trys: %d\n", numTrys)
	fmt.Printf("Error: %f\n", besterr)
	fmt.Println("Trained with:")
	for i := 0; i < len(bestfeats); i++ {
		fmt.Printf("%f,%f\n", bestfeats[i][0], math.Sin(bestfeats[i][0]))
	}
	fmt.Println("\nRandom test:")
	for i := 0; i < 50; i++ {
		x := float64(i) / 50 * math.Pi
		r := n.Predict([]float64{x})
		fmt.Printf("%f,%f\n", x, r[0])
	}
}

func TestBP(t *testing.T) {
	// Create some training data for the sine function on [0,pi]
	numTP := 10
	feats := make([][]float64, numTP)
	targs := make([][]float64, numTP)
	for i := 0; i < numTP; i++ {
		x := float64(i)/10*float64(numTP) - 5.0
		//x := float64(i) / float64(numTP-1) * math.Pi
		feats[i] = []float64{x}
		//targs[i] = []float64{math.Sin(feats[i][0])}
		targs[i] = []float64{activation.Logistic(x)}
	}

	// Create a nn
	n := NewANN(1, 1, activation.Logistic)
	n.AddHidden(25, true, activation.Logistic)
	n.Wire()

	// Train the network with the data
	err := n.Backprop(feats, targs, 200000, 0.02, 1, 0.3)
	log.Printf("Error after training: %f\n", err)

	log.Println("Validation:")
	for i := 0; i < len(feats); i++ {
		fmt.Printf("%f,%f\n", feats[i][0], targs[i][0])
	}
	// test sin
	//for i := 0; i < 50; i++ {
	//	x := float64(i) / 50 * math.Pi
	//	r := n.Predict([]float64{x})
	//	fmt.Printf("%f,,%f\n", x, r[0])
	//}

	//test logistic
	for i := 0; i < 50; i++ {
		x := float64(i)/50*10 - 5
		r := n.Predict([]float64{x})
		fmt.Printf("%f,,%f\n", x, r[0])
	}
}
