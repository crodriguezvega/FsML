namespace FsML.Common.Tests

open FsCheck
open FsCheck.Xunit
open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Statistics

open FsML.Common

module UtilsTests =

    type FeatureVector =
        static member DenseVector () =
            gen {
                let! (mean, variance) = Gen.elements [-1000.0..0.1..1000.0] |> Gen.two
                let normalDistribution = Normal.WithMeanVariance(mean, abs (variance))

                let vector = Array.create 100 0.0
                normalDistribution.Samples vector
                return vector |> DenseVector.OfArray
            } |> Arb.fromGen

    [<Property(Arbitrary=[| typeof<FeatureVector> |])>]
    let ``Can scale vector to zero mean and unit standard deviation`` (x: DenseVector) =
        let epsilon = 0.001
        match Utils.scaling x with
        | None -> false
        | Some scaled -> ((Statistics.Mean scaled) - 0.0) < epsilon && ((Statistics.StandardDeviation scaled) - 1.0) < epsilon
