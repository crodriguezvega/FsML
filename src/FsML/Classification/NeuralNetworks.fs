namespace FsML

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module NeuralNetworks =

  /// Sigmoid function
  let sigmoidFunction (Z: Matrix<float>) =
    1.0 / (1.0 + (-1.0 * Z).PointwiseExp())

  /// Gradient of sigmoid function
  let gradientOfSigmoidFunction (Z: Matrix<float>) =
    let sigZ = sigmoidFunction Z
    sigZ.PointwiseMultiply(1.0 - sigZ)

  /// Cost function
  let costFunction regularization (Z: Matrix<float>) (Y: Matrix<float>) (thetas: Matrix<float> list) numberOfTrainingSamples =
    let aux = 1.0 / float Z.RowCount
    let costPositive = -1.0 * (Y.TransposeAndMultiply((sigmoidFunction Z).PointwiseLog()))
    let costNegative = (1.0 - Y).TransposeAndMultiply((1.0 - sigmoidFunction Z).PointwiseLog())
    let costWithoutRegularization = aux * (costPositive - costNegative).RowSums().Sum()
    match regularization with
    | Optimization.Regularization.Without -> costWithoutRegularization
    | Optimization.Regularization.With(lambda) ->
      let regularizationTerm = [ for theta in thetas -> theta.RemoveColumn(0).PointwisePower(2.0).RowSums().Sum() ] |> List.fold (+) 0.0
      costWithoutRegularization + ((lambda * aux) / (2.0 * (float numberOfTrainingSamples))) * regularizationTerm

  /// Forward propagation
  let rec forwardPropagation (X: Matrix<float>) (thetas: Matrix<float> list): ((Matrix<float> * Matrix<float> option) list) =
    match thetas with
    | [] -> [(X, None)] // Trying to simulate an empty matrix for z
    | theta :: tail ->
      // Add bias unit
      let a = X.InsertColumn(0, DenseVector.Create(X.RowCount, 1.0))
      // Multiply activation with theta
      let z = a.TransposeAndMultiply(theta)
      // Result of sigmoid function becomes new input
      (a, Some(z)) :: (forwardPropagation (sigmoidFunction z) tail)

  /// Gradient of cost function
  let gradientOfCostFunction regularization (delta: Matrix<float>) (a: Matrix<float>) (unbiasedTheta: Matrix<float>) numberOfTrainingSamples =
    let numberOfRow = delta.RowCount
    let rec loop i (acc: Matrix<float>) =
      match i with
      | _ when i = numberOfRow ->
        let aux = 1.0 / (float numberOfTrainingSamples)
        let gradientWithoutRegularization = aux * acc
        match regularization with
        | Optimization.Regularization.Without -> gradientWithoutRegularization
        | Optimization.Regularization.With(lambda) -> 
          let regularizationTerm = lambda * unbiasedTheta
          gradientWithoutRegularization + aux * regularizationTerm
      | _ ->
        let inputRow = [| delta.Row(i) |] |> DenseMatrix.OfColumnVectors
        let aRow = [| a.Row(i) |] |> DenseMatrix.OfRowVectors
        loop (i + 1) (acc + inputRow.Multiply(aRow))
    loop 0 (DenseMatrix.zero delta.ColumnCount a.ColumnCount)

  /// Back propagation
  let rec backPropagation regularization (input: Matrix<float>) (thetas: Matrix<float> list) az numberOfTrainingSamples =
    match thetas, az with
    | _, [] -> []
    | theta :: tailTheta, (aLast, None) :: (aSecondToLast, zSecondToLast) :: tailAz ->
      // Remove elements for bias unit
      let unbiasedTheta = theta.RemoveColumn(0)
      // Calculate error in the activation nodes of the output layer
      let delta = aLast - input
      // Calculate gradient of cost function at the output layer
      let gradient = gradientOfCostFunction regularization delta aSecondToLast unbiasedTheta numberOfTrainingSamples
      (gradient :: backPropagation regularization delta (theta :: tailTheta) tailAz numberOfTrainingSamples)
    | theta :: tailTheta, (a, Some(z)) :: tailAz ->
      // Remove elements for bias unit
      let unbiasedTheta = theta.RemoveColumn(0)
      // Calculate the error in the activation nodes of the hidden layer
      let delta = input.Multiply(unbiasedTheta).PointwiseMultiply(gradientOfSigmoidFunction z)
      // Calculate gradient of cost function at the hidden layer
      let gradient = gradientOfCostFunction regularization delta a unbiasedTheta numberOfTrainingSamples
      gradient :: (backPropagation regularization delta tailTheta tailAz numberOfTrainingSamples)
    | _, _ -> failwith "Length of lists of theta matrices and (a, z) tuples do not match"

  /// Train
  let train regularization (X: Matrix<float>) Y thetas (learningRate: float) numberOfiterations =
    let rec update (thetas: Matrix<float> list) (gradients: Matrix<float> list) =
      match thetas, gradients with
      | [], [] -> []
      | theta :: tailTheta, gradient :: tailGradient ->
        let updatedTheta = theta - learningRate * gradient
        updatedTheta :: (update tailTheta tailGradient)
      | _, _ -> failwith "Length of lists of theta and gradient matrices do not match"

    let rec loop i thetas =
      match i with
      | 0 -> thetas
      | _ ->
        let az = forwardPropagation X thetas
        let gradientOfCostFunction = backPropagation regularization Y (List.rev thetas) (List.rev az) X.RowCount
        loop (i - 1) (update thetas (List.rev gradientOfCostFunction))
    loop numberOfiterations thetas

  /// Predict
  let predict input thetas =
    let az = forwardPropagation input thetas
    fst (List.rev az).Head