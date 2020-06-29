namespace FsML.Algorithms.Regression

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

open FsML.Algorithms
open FsML.Domain.Regression

module LinearRegression =

  /// <summary>
  /// Hypothesis
  /// </summary>
  /// <param name="H">Matrix of observations (observation per row and feature per column) (a.k.a. regressors)</param>
  /// <param name="W">Vector of weights</param>
  /// <returns>Vector of predictions</returns>
  let private hypothesis (H: Regressors) (W: Weights) : Predictions = H * W

  let private constFunction' regularization (H: Regressors) (Y: Regressand) (W: Weights) : MSE = 
    let aux = (1.0 / (2.0 * float H.RowCount))
    let prediction = hypothesis H W
    let costWithoutRegularization = aux * (prediction - Y).PointwisePower(2.0).Sum()
    match regularization with
    | Regularization.Without ->
      costWithoutRegularization
    | Regularization.With(lambda) ->
      let regularizationTerm = aux * lambda * W.SubVector(1, W.Count - 1).PointwisePower(2.0).Sum()
      costWithoutRegularization + regularizationTerm

  /// <summary>
  /// Cost function (Mean Square Error)
  /// </summary>
  /// <remarks>
  /// H: Matrix of observations (observation per row and feature per column) (a.k.a. regressors)
  /// Y: Vector of observed values (a.k.a. regressand)
  /// W: Vector of weights
  /// - Without regularization: cost = (1 / (2 * m)) * (H * W - Y) ^ 2
  /// - With regularization:    cost = (1 / (2 * m)) * ((H * W - Y) ^ 2 + λ * Σᵢ₌₀(W[i] ^ 2)
  /// where m = number of observations
  ///       λ = regularization factor
  /// </remarks>
  /// <param name="regularization">Regularization flag</param>
  /// <param name="costParameters">Parameters for cost calculation</param>
  /// <returns>MSE - Mean Square Error</returns>
  let costFunction regularization costParameters : MSE =
    let H = CostParameters.getH costParameters
    let Y = CostParameters.getY costParameters
    let W = CostParameters.getW costParameters

    constFunction' regularization H Y W

  /// <summary>
  /// Fit with gradient descent
  /// </summary>
  /// <remarks>
  /// - If alpha too small -> slow convergence
  /// - If alpha too large -> may not converge
  /// </remarks>
  /// <param name="regularization">Regularization flag</param>
  /// <param name="gradientDescentParameters">Parameters for gradient descent calculation</param>
  /// <param name="trainingParameters">Parameters for training</param>
  /// <returns>Vector of weights</returns>
  let fitWithGradientDescent regularization gradientDescentParameters trainingParameters : Weights =
    Optimization.gradientDescent regularization gradientDescentParameters hypothesis constFunction' trainingParameters

  /// <summary>
  /// Fit with normal equation
  /// </summary>
  /// <remarks>
  /// H: Matrix of observations (observation per row and feature per column) (a.k.a. regressors)
  /// Y: Vector of observed values (a.k.a. regressand)
  /// W: Vector of weights
  /// - Without regularization: W = inverse(Hᵀ * H) * (Hᵀ) * Y
  /// - With regularization:    W = inverse(Hᵀ * H + λ * reg) * (Hᵀ) * Y
  /// where λ = regularization factor
  /// </remarks>
  /// <param name="regularization">Regularization flag</param>
  /// <param name="trainingParameters">Parameters for training</param>
  /// <returns>Vector of weights</returns>
  let fitWithNormalEquation regularization trainingParameters : Weights =
    let H = TrainingParameters.getH trainingParameters
    let Y = TrainingParameters.getY trainingParameters

    let mainTerm = H.TransposeThisAndMultiply H
    match regularization with
    | Regularization.Without ->
      (mainTerm.Inverse().TransposeAndMultiply H) * Y
    | Regularization.With(lambda) ->
      let regularizationTerm = SparseMatrix.diag H.ColumnCount 1.0
      regularizationTerm.[0, 0] <- 0.0
      ((mainTerm + lambda * regularizationTerm).Inverse().TransposeAndMultiply H) * Y

  /// <summary>
  /// Predict
  /// </summary>
  /// <param name="predictionParameters">Parameters for prediction</param>
  /// <returns>Vector of predictions</returns>
  let predict predictionParameters : Predictions =
    let H = PredictionParameters.getH predictionParameters
    let W = PredictionParameters.getW predictionParameters

    hypothesis H W

  /// <summary>
  /// The R² value describes the amount of variation in the observed values that is explained by the regression line
  /// </summary>
  /// <remarks>
  /// H: Matrix of observations (observation per row and feature per column) (a.k.a. regressors)
  /// Y: Vector of observed values (a.k.a. regressand)
  /// W: Vector of weights
  /// - Mean of observed vales: mean = (1 / m) * Σᵢ₌₀(Y[i])
  /// - Total sum of squares: TSS = Σᵢ₌₀((Y[i] - mean) ^ 2)
  /// - Residual sum of squares: RSS = TSS = Σᵢ₌₀((Y[i] - P[i]) ^ 2)
  /// - R² = 1 - (RSS / TSS)
  /// where m = number of observed values
  ///       P = H * W
  /// </remarks>
  /// <param name="goodnessOfFitParameters">Parameters for goodness of fit calculation</param>
  /// <returns>R² value</returns>
  let Rsquared goodnessOfFitParameters : RSquared =
    let H = GoodnessOfFitParameters.getH goodnessOfFitParameters
    let Y = GoodnessOfFitParameters.getY goodnessOfFitParameters
    let W = GoodnessOfFitParameters.getW goodnessOfFitParameters

    let meanY = Y.Sum() / ((float) Y.Count)
    let totalSumOfSquares = Y.Map(fun y -> (y - meanY) ** 2.0).Sum()

    let prediction = List.ofArray ((hypothesis H W).ToArray())
    let residualSumOfSquares = Y.ToArray()
                               |> List.ofArray
                               |> List.zip prediction
                               |> List.map (fun (y, p) -> (y - p) ** 2.0)
                               |> List.sum

    1.0 - (residualSumOfSquares / totalSumOfSquares)

  /// <summary>
  /// The adjusted R² value 
  /// </summary>
  /// <remarks>
  /// Adjusted R² = 1 - (1 - R²) * ((n - 1) / (n - p - 1))
  /// where p = total number of features (not including the intercept term)
  ///       n = number of observations
  /// </remarks>
  /// <param name="goodnessOfFitParameters">Parameters for goodness of fit calculation</param>
  /// <returns>Adjusted R² value</returns>
  let AdjustedRsquared goodnessOfFitParameters : AdjustedRSquared =
    let H = GoodnessOfFitParameters.getH goodnessOfFitParameters

    let rSquared = Rsquared goodnessOfFitParameters
    let n = (float) H.RowCount
    let p = (float) H.ColumnCount - 1.0
    1.0 - (1.0 - rSquared) * ((n - 1.0) / (n - p - 1.0))