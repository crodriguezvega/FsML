namespace FsML

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module NeuralNetworks =

  /// Sigmoid function
  let sigmoidFunction (Z: Matrix<_>) =
    1.0 / (1.0 + (-1.0 * Z).PointwiseExp())

  /// Sigmoid derivative function
  let derivativeOfSigmoidFunction (Z: Matrix<_>) =
    let sigZ = sigmoidFunction Z
    sigZ.PointwiseMultiply(1.0 - sigZ)

  /// Forward propagation
  let rec forwardPropagation (input: Matrix<_>) (thetas: Matrix<_> list): ((Matrix<_> * Matrix<_>) list) =
    match thetas with
    | [] -> [(input, DenseMatrix.create 1 1 0.0)] // Trying to simulate an empty matrix for z
    | theta :: tail ->
      // Add bias unit
      let a = input.InsertColumn(0, DenseVector.Create(input.RowCount, 1.0))
      // Multiply activation with theta
      let z = a.TransposeAndMultiply(theta)
      // Result of sigmoid function becomes new input
      ((a, z) :: (forwardPropagation (sigmoidFunction z) tail))

  /// Calculate delta
  let calculateDelta (input: Matrix<_>) (a: Matrix<_>) =
    let numberOfRow = input.RowCount
    let rec loop i (delta: Matrix<_>) =
      match i with
      | _ when i = numberOfRow -> delta
      | _ ->
        let inputRow = [| input.Row(i) |] |> DenseMatrix.OfColumnVectors
        let aRow = [| a.Row(i) |] |> DenseMatrix.OfRowVectors
        loop (i + 1) (delta + inputRow.Multiply(aRow))
    loop 0 (DenseMatrix.zero input.ColumnCount a.ColumnCount)

  /// Back propagation
  let rec backPropagation (input: Matrix<_>) (thetas: Matrix<_> list) (az: (Matrix<_> * Matrix<_>) list) =
    match thetas, az with
    | _, [] -> []
    | _, (aLast, zLast) :: (aSecondToLast, zSecondToLast) :: tail when (zLast.ToArray() = (Array2D.zeroCreate 1 1)) ->
      // Calculate delta
      let d = aLast - input
      let delta = calculateDelta d aSecondToLast
      (delta :: backPropagation d thetas tail)
    | theta::tailTheta, (a, z)::tailAz ->
      // Remove elements for bias unit
      let noBiasTheta = theta.RemoveColumn(0)
      // Calculate delta
      let d = input.Multiply(noBiasTheta).PointwiseMultiply(derivativeOfSigmoidFunction z)
      let delta = calculateDelta d a
      (delta :: backPropagation d tailTheta tailAz)
    | _, _ -> [] // Should not get here
