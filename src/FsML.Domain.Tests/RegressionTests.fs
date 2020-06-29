namespace FsML.Common.Tests

open Xunit
open MathNet.Numerics.LinearAlgebra

open FsML.Domain.Types
open FsML.Domain.Regression

module RegressionTests =

  [<Fact>]
  let ``Cannot create training parameters with invalid dimensions``() =

    let H = Matrix<double>.Build.Random(2, 1)
    let Y = Vector<double>.Build.Random(3)

    let result = TrainingParameters.create H Y

    match result with
    | Error ([InvalidDimensions _]) -> Assert.True(true)
    | _ -> Assert.True(false)

  [<Fact>]
  let ``Cannot create prediction parameters with invalid dimensions``() =

    let H = Matrix<double>.Build.Random(2, 1)
    let W = Vector<double>.Build.Random(2)

    let result = PredictionParameters.create H W

    match result with
    | Error ([InvalidDimensions _]) -> Assert.True(true)
    | _ -> Assert.True(false)

  [<Fact>]
  let ``Cannot create goodness of fitness parameters with invalid dimensions``() =

    let H = Matrix<double>.Build.Random(2, 1)
    let Y = Vector<double>.Build.Random(3)
    let W = Vector<double>.Build.Random(2)

    let result = GoodnessOfFitParameters.create H Y W

    match result with
    | Error ([InvalidDimensions _]) -> Assert.True(true)
    | _ -> Assert.True(false)

  [<Fact>]
  let ``Cannot create cost parameters with invalid dimensions``() =

    let H = Matrix<double>.Build.Random(2, 1)
    let Y = Vector<double>.Build.Random(3)
    let W = Vector<double>.Build.Random(2)

    let result = CostParameters.create H Y W

    match result with
    | Error ([InvalidDimensions _]) -> Assert.True(true)
    | _ -> Assert.True(false)