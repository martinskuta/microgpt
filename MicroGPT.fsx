#!/usr/bin/dotnet fsi

open System
open System.Collections.Generic
open System.Globalization
open System.IO
open System.Net.Http
open System.Numerics
open System.Text

type Value(data: float, children: Value[], localGrads: float[]) =
    let mutable _data = data
    let mutable _grad = 0.0

    /// Leaf node constructor (no children)
    new(data: float) = Value(data, [||], [||])

    member _.Data       with get() = _data and set(v) = _data <- v
    member _.Grad       with get() = _grad and set(v) = _grad <- v
    member _.Children   = children
    member _.LocalGrads = localGrads

    static member (+)(a: Value, b: Value) = Value(a.Data + b.Data, [|a; b|], [|1.0; 1.0|])
    static member (+)(a: Value, b: float) = a + Value b
    static member (+)(a: float, b: Value) = Value a + b

    static member (*)(a: Value, b: Value) = Value(a.Data * b.Data, [|a; b|], [|b.Data; a.Data|])
    static member (*)(a: Value, b: float) = a * Value b
    static member (*)(a: float, b: Value) = Value a * b

    static member (~-)(a: Value)          = a * -1.0
    static member (-)(a: Value, b: Value) = a + -b
    static member (-)(a: float, b: Value) = Value a + -b
    static member (-)(a: Value, b: float) = a + -Value b

    static member (/)(a: Value, b: Value) = a * b.Pow(-1.0)
    static member (/)(a: float, b: Value) = Value a * b.Pow(-1.0)
    static member (/)(a: Value, b: float) = a * Math.Pow(b, -1.0)

    member this.Pow(other: float) =
        Value(Math.Pow(_data, other), [|this|], [|other * Math.Pow(_data, other - 1.0)|])

    member this.Log() = Value(Math.Log(_data), [|this|], [|1.0 / _data|])

    member this.Exp() =
        let e = Math.Exp _data
        Value(e, [|this|], [|e|])

    member this.Relu() =
        Value(Math.Max(0.0, _data), [|this|], [|if _data > 0.0 then 1.0 else 0.0|])

    static member Dot(a: ResizeArray<Value>, b: ResizeArray<Value>) =
        let n          = a.Count
        let children   = Array.zeroCreate<Value> (2 * n)
        let localGrads = Array.zeroCreate<float> (2 * n)

        for i in 0 .. n - 1 do
            let va = a.[i]
            let vb = b.[i]
            children.[i]     <- va
            children.[n + i] <- vb
            localGrads.[i]     <- vb.Data   // grad of a_i = b_i
            localGrads.[n + i] <- va.Data   // grad of b_i = a_i

        let mutable dotData  = 0.0
        let vecCount         = Vector<float>.Count
        let vecLoopEnd       = n - vecCount
        let mutable j        = 0

        if vecLoopEnd >= 0 then
            let bDataSpan      = localGrads.AsSpan(0, n)
            let aDataSpan      = localGrads.AsSpan(n, n)
            let mutable sumVec = Vector<float>.Zero

            while j <= vecLoopEnd do
                let va = Vector<float>(aDataSpan.Slice(j, vecCount))
                let vb = Vector<float>(bDataSpan.Slice(j, vecCount))
                sumVec <- sumVec + va * vb
                j      <- j + vecCount

            dotData <- Vector.Sum sumVec

        while j < n do
            dotData <- dotData + localGrads.[n + j] * localGrads.[j]
            j       <- j + 1

        Value(dotData, children, localGrads)

    member this.Backward
        (topo: ResizeArray<Value>, visited: HashSet<Value>,
         stack: Stack<struct (Value * int)>) =

        stack.Push(struct (this, 0))

        while stack.Count > 0 do
            let struct (current, childIndex) = stack.Pop()
            let ch = current.Children

            if ch.Length > 0 && childIndex < ch.Length then
                stack.Push(struct (current, childIndex + 1))
                let child = ch.[childIndex]
                if visited.Add child then
                    stack.Push(struct (child, 0))
            else
                topo.Add current

        this.Grad <- 1.0

        for topoIdx = topo.Count - 1 downto 0 do
            let v     = topo.[topoIdx]
            let vGrad = v.Grad

            if vGrad <> 0.0 then
                let ch = v.Children
                if ch.Length > 0 then
                    let lg  = v.LocalGrads
                    let len = ch.Length

                    match len with
                    | 1 ->
                        ch.[0].Grad <- ch.[0].Grad + lg.[0] * vGrad
                    | 2 ->
                        ch.[0].Grad <- ch.[0].Grad + lg.[0] * vGrad
                        ch.[1].Grad <- ch.[1].Grad + lg.[1] * vGrad
                    | _ ->
                        for i in 0 .. len - 1 do
                            ch.[i].Grad <- ch.[i].Grad + lg.[i] * vGrad

    override _.ToString() = sprintf "Value(data=%f)" _data

let parseArg (args: string[]) (name: string) (defaultVal: 'T) : 'T =
    let mutable result = defaultVal
    let mutable i      = 0
    while i < args.Length - 1 do
        if args.[i].StartsWith "--" && args.[i].[2..] = name then
            result <-
                if   typeof<'T> = typeof<bool>   then box (args.[i+1].Equals("true", StringComparison.OrdinalIgnoreCase)) :?> 'T
                elif typeof<'T> = typeof<int>    then box (int args.[i+1]) :?> 'T
                elif typeof<'T> = typeof<float>  then box (Double.Parse(args.[i+1], CultureInfo.InvariantCulture)) :?> 'T
                elif typeof<'T> = typeof<string> then box args.[i+1] :?> 'T
                else failwithf "Invalid type %A for argument %s" typeof<'T> name
        i <- i + 1
    result

let gauss (rng: Random) mean std =
    let u1 = 1.0 - rng.NextDouble()
    let u2 = 1.0 - rng.NextDouble()
    mean + std * Math.Sqrt(-2.0 * Math.Log u1) * Math.Sin(2.0 * Math.PI * u2)

let createMatrix (rng: Random) nout nin std =
    ResizeArray(Array.init nout (fun _ ->
        ResizeArray(Array.init nin (fun _ -> Value(gauss rng 0.0 std)))))

let linear (x: ResizeArray<Value>) (w: ResizeArray<ResizeArray<Value>>) =
    ResizeArray(w |> Seq.map (fun wo -> Value.Dot(wo, x)))

let softmax (logits: ResizeArray<Value>) =
    let maxVal = logits |> Seq.map (fun v -> v.Data) |> Seq.max
    let exps   = ResizeArray(logits |> Seq.map (fun v -> (v - maxVal).Exp()))
    let total  = exps |> Seq.fold (fun acc e -> acc + e) (Value 0.0)
    ResizeArray(exps |> Seq.map (fun e -> e / total))

let rmsNorm (x: ResizeArray<Value>) =
    let sumSq = x |> Seq.fold (fun acc xi -> acc + xi * xi) (Value 0.0)
    let ms    = sumSq / float x.Count
    let scale = (ms + 1e-5).Pow(-0.5)
    ResizeArray(x |> Seq.map (fun xi -> xi * scale))

let shuffle (rng: Random) (lst: ResizeArray<'T>) =
    let arr = lst.ToArray()
    for i = arr.Length - 1 downto 1 do
        let j   = rng.Next(i + 1)
        let tmp = arr.[i]
        arr.[i] <- arr.[j]
        arr.[j] <- tmp
    ResizeArray arr

// Parse CLI arguments (--n_embd 16)
let args = fsi.CommandLineArgs.[1..]

let nEmbd        = parseArg args "n_embd"         16
let nLayer       = parseArg args "n_layer"          1
let blockSize    = parseArg args "block_size"       8
let numSteps     = parseArg args "num_steps"    10000
let nHead        = parseArg args "n_head"           4
let learningRate = parseArg args "learning_rate"  1e-2
let seed         = parseArg args "seed"            42
let inputUrl     = parseArg args "input_url"
                       "https://raw.githubusercontent.com/martinskuta/microgpt/refs/heads/main/input.txt"
let headDim = nEmbd / nHead

let random = Random seed

// Input dataset

if not (File.Exists "input.txt") then
    printfn "Downloading input.txt..."
    use client = new HttpClient()
    let data   = client.GetStringAsync(inputUrl).GetAwaiter().GetResult()
    File.WriteAllText("input.txt", data)

let docs =
    File.ReadAllLines "input.txt"
    |> Array.map    (fun l -> l.Trim())
    |> Array.filter (fun l -> not (String.IsNullOrEmpty l))
    |> (fun arr -> shuffle random (ResizeArray arr))

printfn "num docs: %d" docs.Count

// Tokenizer
let allChars =
    docs |> Seq.collect id |> Seq.distinct |> Seq.sort |> Array.ofSeq

let bos       = allChars.Length
let vocabSize = allChars.Length + 1
printfn "vocab size: %d" vocabSize

let encode (c: char) = Array.findIndex ((=) c) allChars
let decode (i: int)  = if i = bos then '.' else allChars.[i]

let stateDict = Dictionary<string, ResizeArray<ResizeArray<Value>>>()
stateDict.["wte"]     <- createMatrix random vocabSize  nEmbd       0.08
stateDict.["wpe"]     <- createMatrix random blockSize  nEmbd       0.08
stateDict.["lm_head"] <- createMatrix random vocabSize  nEmbd       0.08

for i in 0 .. nLayer - 1 do
    stateDict.[sprintf "layer%d.attn_wq" i] <- createMatrix random nEmbd        nEmbd       0.08
    stateDict.[sprintf "layer%d.attn_wk" i] <- createMatrix random nEmbd        nEmbd       0.08
    stateDict.[sprintf "layer%d.attn_wv" i] <- createMatrix random nEmbd        nEmbd       0.08
    stateDict.[sprintf "layer%d.attn_wo" i] <- createMatrix random nEmbd        nEmbd       0.08
    stateDict.[sprintf "layer%d.mlp_fc1" i] <- createMatrix random (4 * nEmbd)  nEmbd       0.08
    stateDict.[sprintf "layer%d.mlp_fc2" i] <- createMatrix random nEmbd        (4 * nEmbd) 0.08

let paramsList =
    stateDict.Values |> Seq.collect id |> Seq.collect id |> Seq.toList

printfn "num params: %d" paramsList.Length

// Cache
let topo    = ResizeArray<Value>()
let visited = HashSet<Value>()
let stack   = Stack<struct (Value * int)>()

// Adam Optimizer
let beta1   = 0.85
let beta2   = 0.99
let epsAdam = 1e-8

let mArr = Array.zeroCreate<float> paramsList.Length
let vArr = Array.zeroCreate<float> paramsList.Length

// GPT function
let gpt tokenId posId
        (keys:   ResizeArray<ResizeArray<Value>>[])
        (values: ResizeArray<ResizeArray<Value>>[]) =

    let tokEmb = stateDict.["wte"].[tokenId]
    let posEmb = stateDict.["wpe"].[posId]

    let mutable x =
        ResizeArray(Array.init nEmbd (fun i -> tokEmb.[i] + posEmb.[i]))
    x <- rmsNorm x

    for li in 0 .. nLayer - 1 do
        let xResidual = ResizeArray x
        x <- rmsNorm x

        let q = linear x stateDict.[sprintf "layer%d.attn_wq" li]
        let k = linear x stateDict.[sprintf "layer%d.attn_wk" li]
        let v = linear x stateDict.[sprintf "layer%d.attn_wv" li]

        keys.[li].Add k
        values.[li].Add v

        let xAttn = ResizeArray<Value>()
        for h in 0 .. nHead - 1 do
            let hs = h * headDim
            let qH = q.GetRange(hs, headDim)
            let T  = keys.[li].Count

            let attnLogits = ResizeArray<Value>()
            for t in 0 .. T - 1 do
                let kH      = keys.[li].[t].GetRange(hs, headDim)
                let mutable dot = Value 0.0
                for j in 0 .. headDim - 1 do
                    dot <- dot + qH.[j] * kH.[j]
                attnLogits.Add(dot / Math.Sqrt(float headDim))

            let attnWeights = softmax attnLogits

            let headOut = ResizeArray(Array.init headDim (fun _ -> Value 0.0))
            for t in 0 .. T - 1 do
                let vH = values.[li].[t].GetRange(hs, headDim)
                let w  = attnWeights.[t]
                for j in 0 .. headDim - 1 do
                    headOut.[j] <- headOut.[j] + w * vH.[j]

            xAttn.AddRange headOut

        x <- linear xAttn stateDict.[sprintf "layer%d.attn_wo" li]
        for i in 0 .. nEmbd - 1 do
            x.[i] <- x.[i] + xResidual.[i]

        // MLP
        let xResidual2 = ResizeArray x
        x <- rmsNorm x
        x <- linear x stateDict.[sprintf "layer%d.mlp_fc1" li]
        x <- ResizeArray(x |> Seq.map (fun xi -> xi.Relu()))
        x <- linear x stateDict.[sprintf "layer%d.mlp_fc2" li]
        for i in 0 .. nEmbd - 1 do
            x.[i] <- x.[i] + xResidual2.[i]

    linear x stateDict.["lm_head"]

for step in 0 .. numSteps - 1 do
    let doc = docs.[step % docs.Count]
    let tokens = ResizeArray<int>()
    tokens.Add bos
    tokens.AddRange(doc |> Seq.map encode)
    tokens.Add bos

    let n = min blockSize (tokens.Count - 1)

    // Initialize KV cache
    let keys   = Array.init nLayer (fun _ -> ResizeArray<ResizeArray<Value>>())
    let values = Array.init nLayer (fun _ -> ResizeArray<ResizeArray<Value>>())

    let losses = ResizeArray<Value>()
    for posId in 0 .. n - 1 do
        let tokenId  = tokens.[posId]
        let targetId = tokens.[posId + 1]
        let logits   = gpt tokenId posId keys values
        let probs    = softmax logits
        losses.Add(-probs.[targetId].Log())

    let mutable loss = Value 0.0
    for l in losses do loss <- loss + l
    loss <- loss * (1.0 / float n)

    for p in paramsList do p.Grad <- 0.0
    topo.Clear()
    visited.Clear()
    stack.Clear()
    loss.Backward(topo, visited, stack)

    let lrT = learningRate * (1.0 - float step / float numSteps)
    for i in 0 .. paramsList.Length - 1 do
        let p = paramsList.[i]
        mArr.[i] <- beta1 * mArr.[i] + (1.0 - beta1) * p.Grad
        vArr.[i] <- beta2 * vArr.[i] + (1.0 - beta2) * (p.Grad ** 2.0)
        let mHat = mArr.[i] / (1.0 - beta1 ** float (step + 1))
        let vHat = vArr.[i] / (1.0 - beta2 ** float (step + 1))
        p.Data <- p.Data - lrT * mHat / (Math.Sqrt vHat + epsAdam)

    if (step + 1) % 100 = 0 then
        printfn "step %4d / %4d | loss %.4f" (step + 1) numSteps loss.Data

// Inference
printfn "\n--- inference (new, hallucinated names) ---"
let temperature = 0.5

for sampleIdx in 0 .. 19 do
    let keys   = Array.init nLayer (fun _ -> ResizeArray<ResizeArray<Value>>())
    let values = Array.init nLayer (fun _ -> ResizeArray<ResizeArray<Value>>())

    let mutable tokenId = bos
    let sample          = StringBuilder()
    let mutable posId   = 0
    let mutable stop    = false

    while posId < blockSize && not stop do
        let logits       = gpt tokenId posId keys values
        let scaledLogits = ResizeArray(logits |> Seq.map (fun l -> l / temperature))
        let probs        = softmax scaledLogits

        // Weighted random choice
        let probsData = probs |> Seq.map (fun p -> p.Data) |> Array.ofSeq
        let mutable r = random.NextDouble() * Array.sum probsData

        let mutable sum       = 0.0
        let mutable nextToken = probsData.Length - 1
        let mutable found     = false
        let mutable i         = 0
        while i < probsData.Length && not found do
            sum <- sum + probsData.[i]
            if r <= sum then
                nextToken <- i
                found     <- true
            i <- i + 1

        tokenId <- nextToken
        if tokenId = bos then stop <- true
        else sample.Append(decode tokenId) |> ignore

        posId <- posId + 1

    printfn "sample %2d: %s" (sampleIdx + 1) (string sample)