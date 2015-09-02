package nn

import (
	"log"
	"math"
)

////////////////////////////////////////////////////////////////////////////////
//							Backpropagation Training                          //
//                                                                            //
// ANN training algorithm that uses simple backpropogation to adjust weights  //
// according to the error signal given by:                                    //
//     (expected - prediction) * prediction * (1 - prediction)                //
// Updates are made after every training element, and after each iteration    //
// through all of the training elements the error is computed with the        //
// cost function (errfn).                                                     //
//                                                                            //
// PARAMETERS:                                                                //
//     feat:   Training input set (features). The length of each inner array  //
//             should be equal to the number of input nodes of the ANN, and   //
//             the length of the outer array (size of training set) should    //
//             match the size of the target (targ) outer array.               //
//                                                                            //
//     targ:   Training input set (targets). The length of each inner array   //
//             should be equal to the number of output nodes of the ANN, and  //
//             the length of the outer array (size of training set) should    //
//             match the size of the features (feat) outer array.             //
//                                                                            //
//     maxItr: The maximum number of iterations through the whole training    //
//             set. Training may stop earlier if minErr is attained.          //
//                                                                            //
//     minErr: The target error value for training. Units are percentage      //
//             points (?). Training will stop if the cost function (errfn)    //
//             less than or equal to this value after an iteration through    //
//             all of the training examples. Training may stop early if       //
//             maxItr is reached first.                                       //
//                                                                            //
//     lr:     Learning rate allows for control over how quickly weights      //
//             respond to the error signals. Normally set to 1, but it        //
//             can be increased or decreased to speed up or slow down         //
//             learning.                                                      //
//                                                                            //
//     mf:     Momentum factor [0-1]. Momentum is used when computing the     //
//             adjustement for each weight, and allows the previous           //
//             change to be represented.                                      //
////////////////////////////////////////////////////////////////////////////////
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
			log.Printf("error: %f\n", errfn(this.SaveWeights()))
			return errfn(this.SaveWeights())
		}
	}
	log.Printf("Gaveup after %d iterations.\n", maxItr)
	log.Printf("error: %f\n", errfn(this.SaveWeights()))
	return errfn(this.SaveWeights())
}
