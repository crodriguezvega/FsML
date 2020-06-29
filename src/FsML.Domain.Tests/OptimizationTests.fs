namespace FsML.Common.Tests

open Xunit

open FsML.Domain.Types
open FsML.Domain.Optimization

module OptimizationTests =

  [<Fact>]
  let ``Cannot create zero tolerance``() =
      
    let result = Tolerance.create 0.0

    match result with
    | Error ([InvalidDimensions _]) -> Assert.True(true)
    | _ -> Assert.True(false)

  [<Fact>]
  let ``Cannot create negative tolerance``() =

    let result = Tolerance.create -1.0

    match result with
    | Error ([InvalidDimensions _]) -> Assert.True(true)
    | _ -> Assert.True(false)

  [<Fact>]
  let ``Cannot create zero learning rate``() =

    let result = LearningRate.create 0.0

    match result with
    | Error ([InvalidDimensions _]) -> Assert.True(true)
    | _ -> Assert.True(false)

  [<Fact>]
  let ``Cannot create negative learning rate``() =

    let result = LearningRate.create -1.0

    match result with
    | Error ([InvalidDimensions _]) -> Assert.True(true)
    | _ -> Assert.True(false)

  [<Fact>]
  let ``Cannot create zero number of iterations``() =

    let result = NumberOfIterations.create 0u

    match result with
    | Error ([InvalidDimensions _]) -> Assert.True(true)
    | _ -> Assert.True(false)

  [<Fact>]
  let ``Cannot create gradient descent parameters with invalid parameters``() =

    let result = GradientDescentParameters.create Batch 0.0 0.0 0u

    match result with
    | Error (errors) when (List.length errors = 3) -> Assert.True(true)
    | _ -> Assert.True(false)