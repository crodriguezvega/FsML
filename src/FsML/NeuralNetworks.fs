namespace FsML

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module NeuralNetworks =

  /// Sigmoid function
  let sigmoidFunction (Z: Matrix<_>) =
    1.0 / (1.0 + (-1.0 * Z).PointwiseExp())

  /// Gradient of sigmoid function
  let gradientOfSigmoidFunction (Z: Matrix<_>) =
    let sigZ = sigmoidFunction Z
    sigZ.PointwiseMultiply(1.0 - sigZ)

  /// Cost function
  let costFunction (Z: Matrix<_>) (Y: Matrix<_>) (thetas: Matrix<_> list) numberOfTrainingSamples regularization =
    let aux = 1.0 / float Z.RowCount
    let costPositive = -1.0 * aux * (Y.TransposeAndMultiply((sigmoidFunction Z).PointwiseLog()))
    let costNegative = (1.0 - Y).TransposeAndMultiply((1.0 - sigmoidFunction Z).PointwiseLog())
    let costWithoutRegularization = aux * ((costPositive - costNegative).RowSums().Sum())
    match regularization with
    | Optimization.Regularization.Without -> costWithoutRegularization
    | Optimization.Regularization.With(lambda) ->
      let regularizationTerm = [ for theta in thetas -> theta.RemoveColumn(0).PointwisePower(2.0).RowSums().Sum() ] |> List.fold (+) 0.0
      costWithoutRegularization + ((lambda * aux) / (2.0 * (float numberOfTrainingSamples))) * regularizationTerm

  /// Forward propagation
  let rec forwardPropagation (X: Matrix<_>) (thetas: Matrix<_> list): ((Matrix<_> * Matrix<_>) list) =
    match thetas with
    | [] -> [(X, DenseMatrix.create 1 1 0.0)] // Trying to simulate an empty matrix for z
    | theta :: tail ->
      // Add bias unit
      let a = X.InsertColumn(0, DenseVector.Create(X.RowCount, 1.0))
      // Multiply activation with theta
      let z = a.TransposeAndMultiply(theta)
      // Result of sigmoid function becomes new input
      ((a, z) :: (forwardPropagation (sigmoidFunction z) tail))

  /// Calculate gradient
  let gradientOfCostFunction (delta: Matrix<_>) (a: Matrix<_>) (unbiasedTheta: Matrix<_>) numberOfTrainingSamples regularization =
    let numberOfRow = delta.RowCount
    let rec loop i (acc: Matrix<_>) =
      match i with
      | _ when i = numberOfRow ->
        let aux = 1.0 / (float numberOfTrainingSamples)
        let mainTerm = acc
        match regularization with
        | Optimization.Regularization.Without -> aux * acc
        | Optimization.Regularization.With(lambda) -> 
          let regularizationTerm = lambda * unbiasedTheta
          aux * (acc + regularizationTerm)
      | _ ->
        let inputRow = [| delta.Row(i) |] |> DenseMatrix.OfColumnVectors
        let aRow = [| a.Row(i) |] |> DenseMatrix.OfRowVectors
        loop (i + 1) (acc + inputRow.Multiply(aRow))
    loop 0 (DenseMatrix.zero delta.ColumnCount a.ColumnCount)

  /// Back propagation
  let rec backPropagation (input: Matrix<_>) (thetas: Matrix<_> list) (az: (Matrix<_> * Matrix<_>) list)
                          (numberOfTrainingSamples: int) regularization =
    match thetas, az with
    | _, [] -> []
    | theta :: tailTheta, (aLast, zLast) :: (aSecondToLast, zSecondToLast) :: tailAz when (zLast.ToArray() = (Array2D.zeroCreate 1 1)) ->
      // Remove elements for bias unit
      let unbiasedTheta = theta.RemoveColumn(0)
      // Calculate error in the activation nodes of the output layer
      let delta = aLast - input
      // Calculate gradient of cost function at the output layer
      let gradient = gradientOfCostFunction delta aSecondToLast unbiasedTheta numberOfTrainingSamples regularization
      (gradient :: backPropagation delta (theta :: tailTheta) tailAz numberOfTrainingSamples regularization)
    | theta :: tailTheta, (a, z) :: tailAz ->
      // Remove elements for bias unit
      let unbiasedTheta = theta.RemoveColumn(0)
      // Calculate the error in the activation nodes of the hidden layer
      let delta = input.Multiply(unbiasedTheta).PointwiseMultiply(gradientOfSigmoidFunction z)
      // Calculate gradient of cost function at the hidden layer
      let gradient = gradientOfCostFunction delta a unbiasedTheta numberOfTrainingSamples regularization
      (gradient :: backPropagation delta tailTheta tailAz numberOfTrainingSamples regularization)
    | _, _ -> failwith "Length of lists of theta matrices and (a, z) tuples do not match"

  /// Train
  let train (trainingX: Matrix<_>) (trainingY: Matrix<_>) (thetas: Matrix<_> list) (learningRate: float) numberOfiterations regularization =
    let rec updateTheta (thetas: Matrix<_> list) (gradients: Matrix<_> list) =
      match thetas, gradients with
      | [], [] -> []
      | theta :: tailTheta, gradient :: tailGradient ->
        let updatedTheta = theta - learningRate * gradient
        (updatedTheta :: updateTheta tailTheta tailGradient)
      | _, _ -> failwith "Length of lists of theta and gradient matrices do not match"

    let rec loop i thetas =
      match i with
      | 0 -> thetas
      | _ ->
        let az = forwardPropagation trainingX thetas
        let gradientOfCostFunction = backPropagation trainingY (List.rev thetas) (List.rev az) trainingX.RowCount regularization
        loop (i - 1) (updateTheta thetas (List.rev gradientOfCostFunction))
    loop numberOfiterations thetas

  /// Predict
  let predict (input: Matrix<_>) (thetas: Matrix<_> list) =
    let az = forwardPropagation input thetas
    fst (List.rev az).Head
