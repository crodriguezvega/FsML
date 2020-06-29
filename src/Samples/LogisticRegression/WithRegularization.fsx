#I "/Users/carlosrodriguez-vega/.nuget/packages/xplot.plotly/3.0.1/lib/netstandard2.0"
#I "/Users/carlosrodriguez-vega/.nuget/packages/mathnet.numerics/4.9.0/lib/netstandard2.0"
#I "/Users/carlosrodriguez-vega/.nuget/packages/mathnet.numerics.fsharp/4.9.0/lib/netstandard2.0"

#r "../../../build/Debug/netcoreapp3.0/FsML.Common/FsML.Common.dll"
#r "../../../build/Debug/netcoreapp3.0/FsML.Domain/FsML.Domain.dll"
#r "../../../build/Debug/netcoreapp3.0/FsML.Algorithms/FsML.Algorithms.dll"
#r "XPlot.Plotly.dll"
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"

open XPlot.Plotly
open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

open FsML.Common.Builders
open FsML.Domain.Types
open FsML.Domain.Regression
open FsML.Domain.Optimization
open FsML.Algorithms.Classification.LogisticRegression

module WithRegularization =

  let length = 100
  let normalDistribution = Normal.WithMeanVariance(0.0, 2.0)
  let x1 = [| for i in 1 .. 100 -> normalDistribution.Sample() |]
  let x2 = [| for i in 1 .. 100 -> normalDistribution.Sample() |]

  // Each row is a training sample
  let H = [|
            (Array.create length 1.0) |> Array.toList |> vector
            (x1 |> DenseVector.OfArray).PointwisePower(2.0)
            x2 |> Array.toList |> vector
          |] |> DenseMatrix.OfColumnVectors

  // Each element is the ouput value for each training sample
  let Y = [| for i in 0 .. length - 1 do
             if (H.[i, 2] >= (pown H.[i, 1] 2)) then yield 1.0
             else yield 0.0 |] |> DenseVector.OfArray

  let fit: Result<Weights, ErrorResult list> = Either.either {
    let! gdParameters = GradientDescentParameters.create GradientDescent.Stochastic 0.05 0.05 50000u
    let! trainingParameters = TrainingParameters.create H Y
    let logisticRegressionWithRegularization = fitWithGradientDescent (Regularization.With(0.01)) gdParameters
    let fit = logisticRegressionWithRegularization trainingParameters
    return fit
  }

  let points = (x1, x2) ||> Array.map2 (fun x y -> (x, y))
  let classA = points |> Array.filter (fun x -> (snd x < pown (fst x) 2))
  let classB = points |> Array.filter (fun x -> (snd x >= pown (fst x) 2))
  let x = [| (Array.min x1 - 0.1) .. 0.1 .. (Array.max x1 + 0.1) |] 

  match fit with
  | Error e -> printfn "No fit: %A" e
  | Ok fit -> [
                Scatter(
                  x = (classA |> Array.map fst),
                  y = (classA |> Array.map snd),
                  mode = "markers",
                  name = "Observed values"
                );
                Scatter(
                  x = (classB |> Array.map fst),
                  y = (classB |> Array.map snd),
                  mode = "markers",
                  name = "Observed values"
                );
                Scatter(
                  x = x,
                  y = (x |> Array.map (fun x -> -1.0 * (fit.At(0) + fit.At(1) * (pown x 2)) / fit.At(2))),
                  mode = "lines",
                  name = "Regression line"
                )
              ]
              |> Chart.Plot
              |> Chart.WithWidth 700
              |> Chart.WithHeight 500
              |> Chart.Show