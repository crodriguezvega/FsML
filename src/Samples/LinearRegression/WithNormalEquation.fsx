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
open FsML.Algorithms.Regression.LinearRegression

module WithNormalEquation =

  let normalDistribution = Normal.WithMeanVariance(0.0, 5.0)
  let x = [| 1.0..1.0..10.0 |]
  let y = x |> Array.map (fun x -> (pown x 2) + normalDistribution.Sample())

  // Each row is a training sample
  let H = [|
            (Array.create x.Length 1.0) |> Array.toList |> vector
            (x |> DenseVector.OfArray).PointwisePower(2.0)
          |] |> DenseMatrix.OfColumnVectors

  // Each element is the ouput value for each training sample
  let Y = y |> DenseVector.OfArray

  let fit: Result<Weights, ErrorResult list> = Either.either {
    let! trainingParameters = TrainingParameters.create H Y
    let fit = fitWithNormalEquation Regularization.Without trainingParameters
    return fit
  }

  match fit with
  | Error e -> printfn "No fit: %A" e 
  | Ok fit -> [
                Scatter(
                  x = x,
                  y = y,
                  mode = "markers",
                  name = "Observed values"
                );
                Scatter(
                  x = x,
                  y = (x |> Array.map (fun x -> fit.At(0) + fit.At(1) * (pown x 2))),
                  mode = "lines",
                  name = "Regression line"
                )
              ]
              |> Chart.Plot
              |> Chart.WithWidth 700
              |> Chart.WithHeight 500
              |> Chart.Show