namespace FsML.Algorithms.Regression

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

open FsML.Algorithms
open FsML.Utilities.Types
open FsML.Utilities.Builders

module LinearRegression =

    /// <summary>
    /// Hypothesis
    /// </summary>
    /// <param name="H">Matrix of observations (observation per row and feature per column)</param>
    /// <param name="W">Vector of weights</param>
    let hypothesys (H: Matrix<float>) (W: Vector<float>) =
        if not (H.RowCount > 0 && W.Count > 0 && H.ColumnCount = W.Count) then Error WrongDimensions
        else Ok (H * W)

    /// <summary>
    /// Cost function (Mean Square Error)
    /// </summary>
    /// <remarks>
    /// - Without regularization: cost = (1 / (2 * m)) * (H * W - Y) ^ 2
    /// - With regularization:    cost = (1 / (2 * m)) * ((H * W - Y) ^ 2 + λ * Σ(W[1:end] ^ 2)
    /// where m = number of observations
    ///       λ = regularization factor
    /// </remarks>
    /// <param name="H">Matrix of observations (observation per row and feature per column)</param>
    /// <param name="Y">Vector of observed values</param>
    /// <param name="W">Vector of weights</param>
    /// <returns>MSE</returns>
    let costFunction regularization (H: Matrix<float>) (Y: Vector<float>) (W: Vector<float>) =
        if not (H.RowCount > 0 && Y.Count > 0 && W.Count > 0 && H.RowCount = Y.Count && H.ColumnCount = W.Count) then
            Error WrongDimensions
        else
            let aux = (1.0 / (2.0 * float H.RowCount))
            Either.either {
                let! hypothesysOutput = hypothesys H W
                let costWithoutRegularization = aux * (hypothesysOutput - Y).PointwisePower(2.0).Sum()
                return match regularization with
                      | Optimization.Regularization.Without ->
                          costWithoutRegularization
                      | Optimization.Regularization.With(lambda) ->
                          let regularizationTerm = aux * lambda * W.SubVector(1, W.Count - 1).PointwisePower(2.0).Sum()
                          costWithoutRegularization + regularizationTerm
            }

    /// <summary>
    /// Gradient of cost function
    /// </summary>
    /// <remarks>
    /// - For batch gradient descent:
    ///   - Without regularization: gradient = (1 / m) * (Hᵀ) * (H * W - Y)
    ///   - With regularization:    gradient = (1 / m) * ((Hᵀ) * (H * W - Y) + λ * [0, W[1:end]])
    /// For stochastic gradient descent:
    ///   - Without regularization: gradient = H[0,:] * ((H[0,:] * W)[0] - Y[0])
    ///   - With regularization:    gradient = H[0,:] * ((H[0,:] * W)[0] - Y[0] + λ * [0, W[1:end]])
    /// where m = number of observations
    ///       λ = regularization factor
    /// </remarks>
    /// <param name="H">Matrix of observations (observation per row and feature per column)</param>
    /// <param name="Y">Vector of observed values</param>
    /// <param name="W">Vector of weights</param>
    let gradientOfCostFunction regularization gradientDescent (H: Matrix<float>) (Y: Vector<float>) (W: Vector<float>) =
        if not (H.RowCount > 0 && Y.Count > 0 && W.Count > 0 && H.RowCount = Y.Count && H.ColumnCount = W.Count) then
            Error WrongDimensions
        else
            let calculate aux (gradientWithoutRegularization: Vector<float>) =
                match regularization with
                | Optimization.Regularization.Without ->
                    gradientWithoutRegularization
                | Optimization.Regularization.With(lambda) ->
                    let regularizationTerm = aux * lambda * (Array.append [|0.0|] (W.SubVector(1, W.Count - 1).ToArray()) |> DenseVector.OfArray)
                    gradientWithoutRegularization + regularizationTerm

            match gradientDescent with
            | Optimization.GradientDescent.Batch ->
                let aux = 1.0 / float H.RowCount
                Either.either {
                    let! hypothesysOutput = hypothesys H W
                    let gradientWithoutRegularization = aux * H.TransposeThisAndMultiply(hypothesysOutput - Y)
                    return calculate aux gradientWithoutRegularization
                }
            | Optimization.GradientDescent.Stochastic ->
                let aux = 1.0
                let trainingSample = H.Row(0)
                let outputSample = Y.At(0)
                Either.either {
                    let! hypothesysOutput = hypothesys (DenseMatrix.OfRowVectors([trainingSample])) W
                    let gradientWithoutRegularization = trainingSample.Multiply(hypothesysOutput.At(0) - outputSample)
                    return calculate aux gradientWithoutRegularization
                }

    /// <summary>
    /// Fit with gradient descent
    /// </summary>
    /// <remarks>
    /// - If alpha too small -> slow convergence
    /// - If alpha too large -> may not converge
    /// </remarks>
    /// <param name="H">Matrix of observations (observation per row and feature per column)</param>
    /// <param name="Y">Vector of observed values</param>
    /// <returns>rss</returns>
    let fitWithGradientDescent regularization gradientDescent (H: Matrix<float>) (Y: Vector<float>) =
        Optimization.gradientDescent regularization gradientDescent costFunction gradientOfCostFunction H Y

    /// <summary>
    /// Fit with normal equation
    /// </summary>
    /// <remarks>
    /// - Without regularization: W = inverse(Hᵀ * H) * (Hᵀ) * Y
    /// - With regularization:    W = 
    /// </remarks>
    /// <param name="regularization">Regularization flag</param>
    /// <param name="H">Matrix of observations (observation per row and feature per column)</param>
    /// <param name="Y">Vector of observed values</param>
    /// <returns>Vector of weights</returns>
    let fitWithNormalEquation regularization (H: Matrix<float>) (Y: Vector<float>) =
        let mainTerm = H.TransposeThisAndMultiply H
        match regularization with
        | Optimization.Regularization.Without ->
            Ok ((mainTerm.Inverse().TransposeAndMultiply H) * Y)
        | Optimization.Regularization.With(lambda) ->
            let regularizationTerm = SparseMatrix.diag H.ColumnCount 1.0
            regularizationTerm.[0, 0] <- 0.0
            Ok (((mainTerm + lambda * regularizationTerm).Inverse().TransposeAndMultiply H) * Y)

    /// <summary>
    /// Predict
    /// </summary>
    /// <param name="H">Matrix of observations (observation per row and feature per column)</param>
    /// <param name="W">Vector of weights</param>
    /// <returns>Vector of predictions</returns>
    let predict (H: Matrix<float>) (W: Vector<float>) = hypothesys H W