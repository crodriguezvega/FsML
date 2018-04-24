#I "../../../build/Debug/netstandard2.0"
#n "MathNet.Numerics.FSharp"

#r "FsML.Utilities.dll"
#r "FsML.Algorithms.dll"

open System
open FsML.Utilities
open FsML.Utilities.Builders

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

let normalDistribution = Normal.WithMeanVariance(0.0, 5.0)
let x = [| 1.0 .. 1.0 .. 10.0 |]
let y = x |> Array.map (fun x -> (pown x 2) + normalDistribution.Sample())

// Each row is a training sample
let trainingX = [|
                  (Array.create x.Length 1.0) |> Array.toList |> vector
                  (x |> DenseVector.OfArray).PointwisePower(2.0)
                |] |> DenseMatrix.OfColumnVectors

// Each element is the ouput value for each training sample
let trainingY = y |> DenseVector.OfArray

let fit = LinearRegression.fitWithNormalEquation Optimization.Regularization.Without trainingX trainingY

let a = either {
  let! a = fitWithGradientDescent
  return 
}