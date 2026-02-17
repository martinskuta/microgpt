#!/usr/bin/dotnet run

using System.Globalization;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

// Parse CLI arguments (--n_embd 16)
var nEmbd = ParseArg(args, "n_embd", 16);
var nLayer = ParseArg(args, "n_layer", 1);
var blockSize = ParseArg(args, "block_size", 8);
var numSteps = ParseArg(args, "num_steps", 10000);
var nHead = ParseArg(args, "n_head", 4);
var learningRate = ParseArg(args, "learning_rate", 1e-2);
var seed = ParseArg(args, "seed", 42);
var inputUrl = ParseArg(args, "input_url",
    "https://raw.githubusercontent.com/martinskuta/microgpt/refs/heads/main/input.txt");
var headDim = nEmbd / nHead;

var random = new Random(seed);

// Input dataset
if (!File.Exists("input.txt"))
{
    Console.WriteLine("Downloading input.txt...");
    using var client = new HttpClient();
    var data = await client.GetStringAsync(inputUrl);
    File.WriteAllText("input.txt", data);
}

var docs = File.ReadAllLines("input.txt")
    .Select(l => l.Trim())
    .Where(l => !string.IsNullOrEmpty(l))
    .Shuffle() // Shuffle so the model doesn't see names in alphabetical order
    .ToList();

Console.WriteLine($"num docs: {docs.Count}");

// Tokenizer
var allChars = string.Join("", docs).Distinct().OrderBy(c => c).ToList();
var bos = allChars.Count;
var vocabSize = allChars.Count + 1;
Console.WriteLine($"vocab size: {vocabSize}");

var stateDict = new Dictionary<string, List<List<Value>>>
{
    ["wte"] = CreateMatrix(random, vocabSize, nEmbd),
    ["wpe"] = CreateMatrix(random, blockSize, nEmbd),
    ["lm_head"] = CreateMatrix(random, vocabSize, nEmbd)
};

for (var i = 0; i < nLayer; i++)
{
    stateDict[$"layer{i}.attn_wq"] = CreateMatrix(random, nEmbd, nEmbd);
    stateDict[$"layer{i}.attn_wk"] = CreateMatrix(random, nEmbd, nEmbd);
    stateDict[$"layer{i}.attn_wv"] = CreateMatrix(random, nEmbd, nEmbd);
    stateDict[$"layer{i}.attn_wo"] = CreateMatrix(random, nEmbd, nEmbd);
    stateDict[$"layer{i}.mlp_fc1"] = CreateMatrix(random, 4 * nEmbd, nEmbd);
    stateDict[$"layer{i}.mlp_fc2"] = CreateMatrix(random, nEmbd, 4 * nEmbd);
}

var paramsList = stateDict.Values.SelectMany(m => m).SelectMany(r => r).ToList();
Console.WriteLine($"num params: {paramsList.Count}");

//cache 
var topo = new List<Value>();
var visited = new HashSet<Value>();
var stack = new Stack<(Value, int)>();

// Adam Optimizer
const double beta1 = 0.85;
const double beta2 = 0.99;
const double epsAdam = 1e-8;

var m = new double[paramsList.Count];
var vAdam = new double[paramsList.Count];

for (var step = 0; step < numSteps; step++)
{
    var doc = docs[step % docs.Count];
    var tokens = new List<int> { bos };
    tokens.AddRange(doc.Select(Encode));
    tokens.Add(bos);

    var n = Math.Min(blockSize, tokens.Count - 1);

    // Initialize KV cache
    var keys = new List<List<Value>>[nLayer];
    var values = new List<List<Value>>[nLayer];
    for (var i = 0; i < nLayer; i++)
    {
        keys[i] = [];
        values[i] = [];
    }

    var losses = new List<Value>();
    for (var posId = 0; posId < n; posId++)
    {
        var tokenId = tokens[posId];
        var targetId = tokens[posId + 1];

        var logits = Gpt(tokenId, posId, keys, values);
        var probs = Softmax(logits);
        var lossT = -probs[targetId].Log();
        losses.Add(lossT);
    }

    var loss = new Value(0);
    foreach (var l in losses) loss += l;
    loss *= 1.0 / n;

    Parallel.ForEach(paramsList, p => p.Grad = 0);
    topo.Clear();
    visited.Clear();
    stack.Clear();
    loss.Backward(topo, visited, stack);

    var lrT = learningRate * (1 - (double)step / numSteps);
    for (var i = 0; i < paramsList.Count; i++)
    {
        var p = paramsList[i];
        m[i] = beta1 * m[i] + (1 - beta1) * p.Grad;
        vAdam[i] = beta2 * vAdam[i] + (1 - beta2) * Math.Pow(p.Grad, 2);

        var mHat = m[i] / (1 - Math.Pow(beta1, step + 1));
        var vHat = vAdam[i] / (1 - Math.Pow(beta2, step + 1));

        p.Data -= lrT * mHat / (Math.Sqrt(vHat) + epsAdam);
    }

    if ((step + 1) % 100 == 0) Console.WriteLine($"step {step + 1,4} / {numSteps,4} | loss {loss.Data:F4}");
}

// Inference
Console.WriteLine("\n--- inference (new, hallucinated names) ---");
const double temperature = 0.5;

for (var sampleIdx = 0; sampleIdx < 20; sampleIdx++)
{
    var keys = new List<List<Value>>[nLayer];
    var values = new List<List<Value>>[nLayer];
    for (var i = 0; i < nLayer; i++)
    {
        keys[i] = [];
        values[i] = [];
    }

    var tokenId = bos;
    var sample = new StringBuilder("");

    for (var posId = 0; posId < blockSize; posId++)
    {
        var logits = Gpt(tokenId, posId, keys, values);
        var scaledLogits = logits.Select(l => l / temperature).ToList();
        var probs = Softmax(scaledLogits);

        // Weighted random choice
        var r = random.NextDouble();
        double sum = 0;
        var nextToken = -1;

        var probsData = probs.Select(p => p.Data).ToList();
        var totalProb = probsData.Sum();
        r *= totalProb;

        for (var i = 0; i < probsData.Count; i++)
        {
            sum += probsData[i];
            if (r <= sum)
            {
                nextToken = i;
                break;
            }
        }

        if (nextToken == -1) nextToken = probsData.Count - 1;

        tokenId = nextToken;
        if (tokenId == bos) break;
        sample.Append(Decode(tokenId));
    }

    Console.WriteLine($"sample {sampleIdx + 1,2}: {sample}");
}

return;

// GPT function
List<Value> Gpt(int tokenId, int posId, List<List<Value>>[] keys, List<List<Value>>[] values)
{
    var tokEmb = stateDict["wte"][tokenId];
    var posEmb = stateDict["wpe"][posId];

    var x = new List<Value>();
    for (var i = 0; i < nEmbd; i++) x.Add(tokEmb[i] + posEmb[i]);
    x = RmsNorm(x);

    for (var li = 0; li < nLayer; li++)
    {
        var xResidual = new List<Value>(x);
        x = RmsNorm(x);

        var q = Linear(x, stateDict[$"layer{li}.attn_wq"]);
        var k = Linear(x, stateDict[$"layer{li}.attn_wk"]);
        var v = Linear(x, stateDict[$"layer{li}.attn_wv"]);

        keys[li].Add(k);
        values[li].Add(v);

        var xAttn = new List<Value>();
        for (var h = 0; h < nHead; h++)
        {
            var hs = h * headDim;
            var qH = q.GetRange(hs, headDim);

            var attnLogits = new List<Value>();
            var T = keys[li].Count;

            for (var t = 0; t < T; t++)
            {
                var kH = keys[li][t].GetRange(hs, headDim);
                var dot = new Value(0);
                for (var j = 0; j < headDim; j++) dot += qH[j] * kH[j];
                attnLogits.Add(dot / Math.Sqrt(headDim));
            }

            var attnWeights = Softmax(attnLogits);

            var headOut = new List<Value>();
            for (var j = 0; j < headDim; j++) headOut.Add(new Value(0));

            for (var t = 0; t < T; t++)
            {
                var vH = values[li][t].GetRange(hs, headDim);
                var w = attnWeights[t];
                for (var j = 0; j < headDim; j++) headOut[j] += w * vH[j];
            }

            xAttn.AddRange(headOut);
        }

        x = Linear(xAttn, stateDict[$"layer{li}.attn_wo"]);
        for (var i = 0; i < nEmbd; i++) x[i] += xResidual[i];

        // MLP
        xResidual = new List<Value>(x);
        x = RmsNorm(x);
        x = Linear(x, stateDict[$"layer{li}.mlp_fc1"]);
        x = x.Select(xi => xi.Relu()).ToList();
        x = Linear(x, stateDict[$"layer{li}.mlp_fc2"]);
        for (var i = 0; i < nEmbd; i++) x[i] += xResidual[i];
    }

    return Linear(x, stateDict["lm_head"]);
}

char Decode(int i) => i == bos ? '.' : allChars[i];

int Encode(char c) => allChars.IndexOf(c);

static double Gauss(Random random, double mean, double std)
{
    var u1 = 1.0 - random.NextDouble();
    var u2 = 1.0 - random.NextDouble();
    var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    return mean + std * randStdNormal;
}

static List<List<Value>> CreateMatrix(Random random, int nout, int nin, double std = 0.08)
{
    var mat = new List<List<Value>>();
    for (var i = 0; i < nout; i++)
    {
        var row = new List<Value>();
        for (var j = 0; j < nin; j++) row.Add(new Value(Gauss(random, 0, std)));

        mat.Add(row);
    }

    return mat;
}

static List<Value> Linear(List<Value> x, List<List<Value>> w) => w.Select(wo => Value.Dot(wo, x)).ToList();

static List<Value> Softmax(List<Value> logits)
{
    var maxVal = logits.Max(v => v.Data);
    var exps = logits.Select(v => (v - maxVal).Exp()).ToList();
    var total = new Value(0);
    foreach (var e in exps) total += e;
    return exps.Select(e => e / total).ToList();
}

static List<Value> RmsNorm(List<Value> x)
{
    var sumSq = new Value(0);
    foreach (var xi in x) sumSq += xi * xi;
    var ms = sumSq / x.Count;
    var scale = (ms + 1e-5).Pow(-0.5);
    return x.Select(xi => xi * scale).ToList();
}

T ParseArg<T>(string[] args, string name, T defaultVal)
{
    for (var i = 0; i < args.Length - 1; i++)
        if (args[i].StartsWith("--") && args[i].AsSpan()[2..] == name)
        {
            if (typeof(T) == typeof(bool))
                return (T)(object)args[i + 1].Equals("true", StringComparison.OrdinalIgnoreCase);
            if (typeof(T) == typeof(int)) return (T)(object)int.Parse(args[i + 1]);
            if (typeof(T) == typeof(double)) return (T)(object)double.Parse(args[i + 1], CultureInfo.InvariantCulture);
            if (typeof(T) == typeof(string)) return (T)(object)args[i + 1];
            throw new ArgumentException($"Invalid type {typeof(T)} for argument {name}");
        }

    return defaultVal;
}

public class Value(double data, Value[]? children = null, double[]? localGrads = null)
{
    private readonly Value[]? _children = children;
    private readonly double[]? _localGrads = localGrads;
    public double Data = data;
    public double Grad;

    public static Value operator +(Value a, Value b) => new(a.Data + b.Data, [a, b], [1.0, 1.0]);

    public static Value operator +(Value a, double b) => a + new Value(b);

    public static Value operator +(double a, Value b) => new Value(a) + b;

    public static Value operator *(Value a, Value b) => new(a.Data * b.Data, [a, b], [b.Data, a.Data]);

    public static Value operator *(Value a, double b) => a * new Value(b);

    public static Value operator *(double a, Value b) => new Value(a) * b;

    public static Value operator -(Value a) => a * -1;

    public static Value operator -(Value a, Value b) => a + -b;

    public static Value operator -(double a, Value b) => a + -b;

    public static Value operator -(Value a, double b) => a + -b;

    public static Value operator /(Value a, Value b) => a * b.Pow(-1);

    public static Value operator /(double a, Value b) => a * b.Pow(-1);

    public static Value operator /(Value a, double b) => a * Math.Pow(b, -1);

    public Value Pow(double other) => new(Math.Pow(Data, other), [this], [other * Math.Pow(Data, other - 1)]);

    public Value Log() => new(Math.Log(Data), [this], [1.0 / Data]);

    public Value Exp() => new(Math.Exp(Data), [this], [Math.Exp(Data)]);

    public Value Relu() => new(Math.Max(0, Data), [this], [Data > 0 ? 1.0 : 0.0]);

    public static Value Dot(List<Value> a, List<Value> b)
    {
        var n = a.Count;

        var children = new Value[2 * n];
        var localGrads = new double[2 * n];

        var aSpan = CollectionsMarshal.AsSpan(a);
        var bSpan = CollectionsMarshal.AsSpan(b);

        for (var i = 0; i < n; i++)
        {
            var va = aSpan[i];
            var vb = bSpan[i];

            children[i] = va;
            children[n + i] = vb;

            // Gradient of (a*b) w.r.t a is b; w.r.t b is a
            localGrads[i] = vb.Data;
            localGrads[n + i] = va.Data;
        }

        double dotData = 0;
        var vecCount = Vector<double>.Count;
        var vecLoopEnd = n - vecCount;
        var j = 0;

        if (vecLoopEnd >= 0)
        {
            var bDataSpan = localGrads.AsSpan(0, n);
            var aDataSpan = localGrads.AsSpan(n, n);
            var sumVector = Vector<double>.Zero;

            for (; j <= vecLoopEnd; j += vecCount)
            {
                var va = new Vector<double>(aDataSpan.Slice(j, vecCount));
                var vb = new Vector<double>(bDataSpan.Slice(j, vecCount));
                sumVector += va * vb;
            }

            dotData = Vector.Sum(sumVector);
        }

        // Remainder scalar loop
        for (; j < n; j++) dotData += localGrads[n + j] * localGrads[j];

        return new Value(dotData, children, localGrads);
    }

    public void Backward(List<Value> topo, HashSet<Value> visited, Stack<(Value node, int childIndex)> stack)
    {
        // Iterative topological sort (avoids recursion overhead)
        stack.Push((this, 0));

        while (stack.Count > 0)
        {
            var (current, childIndex) = stack.Pop();
            var children = current._children;

            if (children != null && childIndex < children.Length)
            {
                stack.Push((current, childIndex + 1));

                var child = children[childIndex];
                if (visited.Add(child))
                {
                    stack.Push((child, 0));
                }
            }
            else
            {
                topo.Add(current);
            }
        }

        Grad = 1.0;

        // Iterate backwards instead of reversing the list
        var topoSpan = CollectionsMarshal.AsSpan(topo);

        // Iterate backwards
        for (var topoIdx = topoSpan.Length - 1; topoIdx >= 0; topoIdx--)
        {
            var v = topoSpan[topoIdx];
            var vGrad = v.Grad;

            // Micro-opt: If gradient is 0, propagation contributes nothing. 
            // Useful for sparse graphs or ReLU dead neurons.
            if (vGrad == 0) continue;

            var children = v._children;
            if (children == null) continue;

            var localGrads = v._localGrads;
            var len = children.Length;

            switch (len)
            {
                // Optimization 2: Manual Loop Unrolling
                // 99% of nodes (Add, Mul, Relu, Pow) have 1 or 2 children.
                // A for-loop has setup and branch prediction overhead.
                case 1:
                    children[0].Grad += localGrads![0] * vGrad;
                    break;
                case 2:
                    children[0].Grad += localGrads![0] * vGrad;
                    children[1].Grad += localGrads[1] * vGrad;
                    break;
                default:
                {
                    for (var i = 0; i < len; i++)
                        children[i].Grad += localGrads![i] * vGrad;
                    break;
                }
            }
        }
    }

    public override string ToString() => $"Value(data={Data})";
}