#I "/Users/carlosrodriguez-vega/.nuget/packages/xplot.plotly/3.0.1/lib/netstandard2.0"
#I "/Users/carlosrodriguez-vega/.nuget/packages/mathnet.numerics/4.9.0/lib/netstandard2.0"
#I "/Users/carlosrodriguez-vega/.nuget/packages/mathnet.numerics.fsharp/4.9.0/lib/netstandard2.0"

#r "../../../build/Debug/netcoreapp3.0/FsML.Algorithms/FsML.Algorithms.dll"
#r "../../../build/Debug/netcoreapp3.0/FsML.Common/FsML.Common.dll"
#r "XPlot.Plotly.dll"
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"

open XPlot.Plotly
open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

open FsML.Algorithms
open FsML.Algorithms.Regression.LinearRegression
open FsML.Common.Builders
open FsML.Common.Types

module WithNormalEquation =

  let normalDistribution = Normal.WithMeanVariance(0.0, 5.0)
  let x = [| 1.0..1.0..10.0 |]
  let y = x |> Array.map (fun x -> (pown x 2) + normalDistribution.Sample())

  // Each row is a training sample
  let trainingX = [|
                    (Array.create x.Length 1.0) |> Array.toList |> vector
                    (x |> DenseVector.OfArray).PointwisePower(2.0)
                  |] |> DenseMatrix.OfColumnVectors

  // Each element is the ouput value for each training sample
  let trainingY = y |> DenseVector.OfArray

  let fit: Result<Vector<float>, ErrorResult> = Either.either {
    let! fit = fitWithNormalEquation Optimization.Regularization.Without trainingX trainingY
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