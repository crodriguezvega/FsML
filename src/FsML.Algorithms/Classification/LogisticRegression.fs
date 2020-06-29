namespace FsML.Algorithms.Classification

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

open FsML.Algorithms
open FsML.Domain.Regression

module LogisticRegression =
    
  /// <summary>
  /// Sigmoid function
  /// </summary>
  /// <param name="Z">Z</param>
  /// <returns>Vector of predictions</returns>
  let sigmoidFunction (Z: Vector<float>) = 1.0 / (1.0 + (-1.0 * Z).PointwiseExp())

  /// <summary>
  /// Sigmoid function
  /// </summary>
  /// <param name="H">Matrix of observations (observation per row and feature per column) (a.k.a. regressors)</param>
  /// <param name="W">Vector of weights</param>
  /// <returns>The estimated probability that y = 1 on input x)</returns>
  let private hypothesis (H: Regressors) (W: Weights) : Predictions = sigmoidFunction (H * W)

  let private constFunction' regularization (H: Regressors) (Y: Regressand) (W: Weights) : MSE = 
    let aux = 1.0 / float H.RowCount
    let costWithoutRegularization = -1.0 * aux * (Y * (sigmoidFunction (H * W)).PointwiseLog()
                                    + (1.0 - Y) * (1.0 - sigmoidFunction (H * W)).PointwiseLog())
    match regularization with
    | Regularization.Without ->
      costWithoutRegularization
    | Regularization.With(lambda) ->
      let regularizationTerm = ((lambda * aux) / 2.0) * W.SubVector(1, W.Count - 1).PointwisePower(2.0).Sum()
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
  /// - If the learning rate is too small -> slow convergence
  /// - If the learning rate is too large -> may not converge
  /// </remarks>
  /// <param name="regularization">Regularization flag</param>
  /// <param name="gradientDescentParameters">Parameters for gradient descent calculation</param>
  /// <param name="trainingParameters">Parameters for training</param>
  /// <returns>Vector of weights</returns>
  let fitWithGradientDescent regularization gradientDescentParameters trainingParameters : Weights =
    Optimization.gradientDescent regularization gradientDescentParameters hypothesis constFunction' trainingParameters

  /// <summary>
  /// Predict
  /// </summary>
  /// <param name="predictionParameters">Parameters for prediction</param>
  /// <returns>Vector of predictions</returns>
  let predict predictionParameters : Predictions =
    let H = PredictionParameters.getH predictionParameters
    let W = PredictionParameters.getW predictionParameters

    Vector.map (fun x -> if x > 0.5 then 1.0 else 0.0 ) (hypothesis H W)