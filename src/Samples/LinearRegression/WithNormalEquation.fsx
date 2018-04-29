#I "../../../build/Debug/FsML.Algorithms"
#I "../../../build/Debug/FsML.Utilities"
#I "../../../packages/FSharp.Charting.0.91.1/lib/net45"
#I "../../../packages/MathNet.Numerics.4.1.0/lib/net40"
#I "../../../packages/MathNet.Numerics.FSharp.4.1.0/lib/net45"

#r "FsML.Algorithms.dll"
#r "FsML.Utilities.dll"
#r "FSharp.Charting.dll"
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"

open System
open FSharp.Charting
open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

open FsML.Utilities
open FsML.Algorithms
open FsML.Algorithms.Regression

// module WithNormalEquation = fsi.AddPrinter(fun (ch:FSharp.Charting.ChartTypes.GenericChart) -> ch.ShowChart() |> ignore; "(Chart)")

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



type TraceBuilder() =
    member this.Bind(m, f) = 
        match m with 
        | None -> 
            printfn "Binding with None. Exiting."
        | Some a -> 
            printfn "Binding with Some(%A). Continuing" a
        Option.bind f m

    member this.Return(x) = 
        Some x

    member this.ReturnFrom(x) = 
        x
        
    member this.Zero() = 
        printfn "Zero"
        this.Return ()

    member this.Delay(f) = 
        printfn "Delay"
        f

    member this.Run(f) = 
        f()

    member this.While(guard, body: unit -> int option) =
        printfn "While: test"
        if not (guard()) 
        then 
            printfn "While: zero"
            this.Zero() 
        else
            printfn "While: body"
            this.Bind( body (), fun () -> 
                this.While(guard, body))  

// make an instance of the workflow  
let trace = new TraceBuilder()

let mutable i = 1
let test() = i < 5
let inc b =
  b <- b + 1

let m = trace {
  while test() do
    printfn "i is %i" i
    inc n
}


type EitherBuilder () =
  member this.Bind(x, f) =
    match x with
    | Ok s -> f s
    | Error f -> Error f
  member this.Return x = Ok(x)
  member this.ReturnFrom x = x
  member this.Zero() = Ok()
  member this.Delay(f) = f
  member this.Combine(x, f) = this.Bind(x, f)
  member this.Run(f) = f()
  member this.While(guard, body) =
    if guard () then this.Bind(body (), fun _ -> this.While(guard, body))
    else this.Zero()

let either = EitherBuilder ()

let mutable i = 0

let func i =
  if i < 3 then Ok(i)
  else Error("error")

let result = either {
  let mutable res = 0
  while i < 10 do 
    let! output = func i
    res <- output
    i <- i + 1
  return res
}

type OptionBuilder() = 
  member x.Bind(v, f) =
    printfn "bind %A" v
    Option.bind f v
  member x.Return(v) = Some v
  member x.ReturnFrom(v) = v
  member x.Zero() = Some ()
  member x.Combine(v, f:unit -> _) = Option.bind f v
  member x.Delay(f : unit -> 'T) = f
  member x.Run(f) = f()
  member x.While(cond, f) =
    if cond() then x.Bind(f(), fun _ -> x.While(cond, f)) 
    else x.Zero()

let maybe = OptionBuilder()
let mutable a = Some 3
// As usual, the type of 'res' is 'Option<int>'
let res = maybe { 
    // The whole body is passed to `Delay` and then to `Run`
    let mutable a = Some 3
    let b = ref 0
    while !b < 10 do 
      let! n = None // This body will be delayed & passed to While
      a <- Some n
      incr b
    // Code following `if` is delayed and passed to Combine
    return! a }


type EitherBuilder () =
  member this.Bind(x, f) =
    printfn "Bind %A" x
    match x with
    | Ok s -> f s
    | Error f -> Error f
  member this.Return x = 
    printfn "Return"
    Ok(x)
  member this.ReturnFrom x =
    printfn "ReturnFrom"
    x
  member this.Zero() =
    printfn "Zero"
    Ok()
  member this.Delay(f) = 
    printfn "Delay %A" f
    f
  member this.Combine(x, f: unit -> _) =
    printfn "Combine"
    this.Bind(x, f)
  member this.Run(f) = f()
  member this.While(guard, body) =
    if guard () then this.Bind(body (), fun _ -> this.While(guard, body))
    else this.Zero()

let either = EitherBuilder ()

let mutable i = 10
let f n =
  if n > 5 then Ok(n)
  else Error("bad")

let mutable b = 0
let guard () = i > 0

let y = either {
  while guard() do
    let! a = f i
    b <- a
    i <- i - 1
  return b
}






let divideBy bottom top =
    if bottom = 0
    then None
    else Some(top/bottom)

let fit = Builders.either {
  let! weights = LinearRegression.fitWithNormalEquation Optimization.Regularization.Without trainingX trainingY
  let! prediction = LinearRegression.predict trainingX weights
  return prediction
}

/// SCALING TEST
let p = vector([ 1.0; 2.0 ])
let o = Optimization.scaling p
match o with
| None -> printfn "None"
| Some(x) -> printfn "%A" x

Chart.Combine(
  [ Chart.Point ((x, y) ||> Array.map2 (fun x y -> (x, y)))
    Chart.Line (x |> Array.map (fun x -> (x, fit.At(0) + fit.At(1) * (pown x 2)))) ])