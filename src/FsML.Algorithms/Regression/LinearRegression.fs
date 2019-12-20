namespace FsML.Algorithms.Regression

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

open FsML.Algorithms
open FsML.Common.Types

module LinearRegression =

  /// <summary>
  /// Hypothesis
  /// </summary>
  /// <param name="H">Matrix of observations (observation per row and feature per column) (a.k.a. regressors)</param>
  /// <param name="W">Vector of weights</param>
  /// <returns>Vector of predictions</returns>
  let private hypothesys (H: Regressors) (W: Weights) : Predictions = H * W

  /// <summary>
  /// Cost function (Mean Square Error)
  /// </summary>
  /// <remarks>
  /// - Without regularization: cost = (1 / (2 * m)) * (H * W - Y) ^ 2
  /// - With regularization:    cost = (1 / (2 * m)) * ((H * W - Y) ^ 2 + λ * Σᵢ₌₀(W[i] ^ 2)
  /// where m = number of observations
  ///       λ = regularization factor
  /// </remarks>
  /// <param name="H">Matrix of observations (observation per row and feature per column) (a.k.a. regressors)</param>
  /// <param name="Y">Vector of observed values (a.k.a. regressand)</param>
  /// <param name="W">Vector of weights</param>
  /// <returns>MSE</returns>
  let costFunction regularization (H: Regressors) (Y: Regressand) (W: Weights) : Result<MSE, ErrorResult> =
    if H.RowCount <> Y.Count then
      Error (InvalidDimensions "The number of observations and observed values must be the same")
    elif H.ColumnCount <> W.Count then
      Error (InvalidDimensions "The number of observations and weights must be the same")
    else
      let aux = (1.0 / (2.0 * float H.RowCount))
      let prediction = hypothesys H W
      let costWithoutRegularization = aux * (prediction - Y).PointwisePower(2.0).Sum()
      Ok (match regularization with
          | Optimization.Regularization.Without ->
            costWithoutRegularization
          | Optimization.Regularization.With(lambda) ->
            let regularizationTerm = aux * lambda * W.SubVector(1, W.Count - 1).PointwisePower(2.0).Sum()
            costWithoutRegularization + regularizationTerm)

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
  /// <param name="H">Matrix of observations (observation per row and feature per column) (a.k.a. regressors)</param>
  /// <param name="Y">Vector of observed values (a.k.a. regressand)</param>
  /// <param name="W">Vector of weights</param>
  let private gradientOfCostFunction regularization gradientDescent (H: Regressors) (Y: Regressand) (W: Weights) : Vector<float> =
    let calculate aux (gradientWithoutRegularization: Vector<float>) =
      match regularization with
      | Optimization.Regularization.Without ->
        gradientWithoutRegularization
      | Optimization.Regularization.With(lambda) ->
        let regularizationTerm = aux * lambda * (Array.append [|0.0|] (W.SubVector(1, W.Count - 1).ToArray()) |> DenseVector.ofArray)
        gradientWithoutRegularization + regularizationTerm

    match gradientDescent with
    | Optimization.GradientDescent.Batch ->
      let aux = 1.0 / float H.RowCount
      let prediction = hypothesys H W
      let gradientWithoutRegularization = aux * H.TransposeThisAndMultiply(prediction - Y)
      calculate aux gradientWithoutRegularization
    | Optimization.GradientDescent.Stochastic ->
      let aux = 1.0
      let trainingSample = H.Row(0)
      let outputSample = Y.At(0)
      let prediction = hypothesys (DenseMatrix.ofRows([trainingSample])) W
      let gradientWithoutRegularization = trainingSample.Multiply(prediction.At(0) - outputSample)
      calculate aux gradientWithoutRegularization

  /// <summary>
  /// Fit with gradient descent
  /// </summary>
  /// <remarks>
  /// - If alpha too small -> slow convergence
  /// - If alpha too large -> may not converge
  /// </remarks>
  /// <param name="H">Matrix of observations (observation per row and feature per column) (a.k.a. regressors)</param>
  /// <param name="Y">Vector of observed values (a.k.a. regressand)</param>
  /// <returns>Vector of weights</returns>
  let fitWithGradientDescent regularization gradientDescent (H: Regressors) (Y: Regressand) : Result<Weights, ErrorResult> =
    if H.RowCount <> Y.Count then
      Error (InvalidDimensions "The number of observations and observed values must be the same")
    else
      Optimization.gradientDescent regularization gradientDescent costFunction gradientOfCostFunction H Y

  /// <summary>
  /// Fit with normal equation
  /// </summary>
  /// <remarks>
  /// - Without regularization: W = inverse(Hᵀ * H) * (Hᵀ) * Y
  /// - With regularization:    W = inverse(Hᵀ * H + λ * reg) * (Hᵀ) * Y
  /// where λ = regularization factor
  /// </remarks>
  /// <param name="regularization">Regularization flag</param>
  /// <param name="H">Matrix of observations (observation per row and feature per column) (a.k.a. regressors)</param>
  /// <param name="Y">Vector of observed values (a.k.a. regressand)</param>
  /// <returns>Vector of weights</returns>
  let fitWithNormalEquation regularization (H: Regressors) (Y: Regressand) : Result<Weights, ErrorResult> =
    if H.RowCount <> Y.Count then
      Error (InvalidDimensions "The number of observations and observed values must be the same")
    else
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
  /// <param name="H">Matrix of observations (observation per row and feature per column) (a.k.a. regressors)</param>
  /// <param name="W">Vector of weights</param>
  /// <returns>Vector of predictions</returns>
  let predict (H: Regressors) (W: Weights) : Result<Predictions, ErrorResult> =
    if H.ColumnCount <> W.Count then
      Error (InvalidDimensions "The number of observations and weights must be the same")
    else
      Ok (hypothesys H W)

  /// <summary>
  /// The R² value describes the amount of variation in the observed values that is explained by the regression line
  /// </summary>
  /// <remarks>
  /// - Mean of observed vales: mean = (1 / m) * Σᵢ₌₀(Y[i])
  /// - Total sum of squares: TSS = Σᵢ₌₀((Y[i] - mean) ^ 2)
  /// - Residual sum of squares: RSS = TSS = Σᵢ₌₀((Y[i] - P[i]) ^ 2)
  /// - R² = 1 - (RSS / TSS)
  /// where m = number of observed values
  ///       P = H * W
  /// </remarks>
  /// <param name="H">Matrix of observations (observation per row and feature per column) (a.k.a. regressors)</param>
  /// <param name="Y">Vector of observed values (a.k.a. regressand)</param>
  /// <param name="W">Vector of weights</param>
  /// <returns>R² value</returns>
  let Rsquared (H: Regressors) (Y: Regressand) (W: Weights) : Result<RSquared, ErrorResult> =
    if H.RowCount <> Y.Count then
      Error (InvalidDimensions "The number of observations and observed values must be the same")
    elif H.ColumnCount <> W.Count then
      Error (InvalidDimensions "The number of observations and weights must be the same")
    else
      let meanY = Y.Sum() / ((float) Y.Count)
      let totalSumOfSquares = Y.Map(fun y -> (y - meanY) ** 2.0).Sum()

      let prediction = List.ofArray ((hypothesys H W).ToArray())
      let residualSumOfSquares = Y.ToArray()
                                 |> List.ofArray
                                 |> List.zip prediction
                                 |> List.map (fun (y, p) -> (y - p) ** 2.0)
                                 |> List.sum

      Ok (1.0 - (residualSumOfSquares / totalSumOfSquares))

  /// <summary>
  /// The adjusted R² value 
  /// </summary>
  /// <remarks>
  /// Adjusted R² = 1 - (1 - R²) * ((n - 1) / (n - p - 1))
  /// where p = total number of features (not including the intercept term)
  ///       n = number of observations
  /// </remarks>
  /// <param name="H">Matrix of observations (observation per row and feature per column) (a.k.a. regressors)</param>
  /// <param name="Y">Vector of observed values (a.k.a. regressand)</param>
  /// <param name="W">Vector of weights</param>
  /// <returns>Adjusted R² value</returns>
  let AdjustedRsquared (H: Regressors) (Y: Regressand) (W: Weights) : Result<AdjustedRSquared, ErrorResult> =
    let rSquared = Rsquared H Y W
    match rSquared with
    | Error e -> Error e
    | Ok rSquared -> 
      let n = (float) H.RowCount
      let p = (float) H.ColumnCount - 1.0
      Ok (1.0 - (1.0 - rSquared) * ((n - 1.0) / (n - p - 1.0)))