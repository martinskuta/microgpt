## Overview

This repository contains two single-file implementations of Andrej Karpathy's **microgpt** — a minimal GPT-style
language model built entirely from scratch, with no ML framework dependencies:

| File | Language | Runtime |
|---|---|---|
| `MicroGPT.cs` | C# 13 | `dotnet run` |
| `MicroGPT.fsx` | **F# 9** | `dotnet fsi` |

The C# version was the original translation from Python, written to explore the algorithm in a familiar .NET stack.
The F# version was then derived from the C# version as a natural evolution toward a more expressive, functional style.

## About the Original

The original microgpt is a minimal implementation of a GPT-style language model from scratch, created by Andrej
Karpathy. It serves as an educational tool to understand the core mechanics of transformer-based language models.

For detailed information about the algorithm and its implementation, please
visit: https://karpathy.github.io/2026/02/12/microgpt/

## Key Features of the Algorithm

microgpt implements a character-level language model using the Transformer architecture. The key components include:

- **Character-level tokenization**: The model operates directly on characters rather than subword tokens
- **Multi-head self-attention**: Enables the model to focus on different parts of the input sequence simultaneously
- **Position embeddings**: Provides the model with information about the position of tokens in the sequence
- **Feed-forward layers**: Adds non-linear transformations to enhance the model's expressiveness
- **RMS normalization**: Stabilizes training by normalizing activations
- **Residual connections**: Helps with gradient flow during training

The model is trained to predict the next character in a sequence, learning patterns and structure from the training
text.

## Prerequisites

- **.NET 10** (or later) must be installed on your system

## How to Run

### C# version

```bash
dotnet run MicroGPT.cs
```

On Unix systems you can also make the file executable and run it directly:

```bash
chmod +x MicroGPT.cs
./MicroGPT.cs
```

### F# version

```bash
dotnet fsi MicroGPT.fsx
```

Both versions accept the same CLI arguments:

```
--n_embd 16 --n_layer 1 --block_size 8 --num_steps 10000
--n_head 4 --learning_rate 0.01 --seed 42
```

---

## F# version — differences and advantages

### Why F#?

F# is a functional-first language that runs on .NET, sharing the same runtime and standard library as C#. Translating
this project to F# illustrates how the same algorithm can be expressed more concisely and with stronger compile-time
guarantees.

### Key differences from the C# version

| Aspect | C# | F# |
|---|---|---|
| **Script execution** | `#!/usr/bin/dotnet run` shebang in a `.cs` file | Standard `dotnet fsi` F# script (`.fsx`) |
| **Entry point** | Top-level statements (C# 9+) | Top-level `let` bindings — idiomatic F# |
| **Mutable state** | `var` everywhere | Explicit `mutable` keyword — immutability is the default |
| **Value class** | Primary constructor syntax | Type with explicit `member` definitions |
| **Collections** | `List<T>` | `ResizeArray<T>` (same BCL type, idiomatic F# alias) |
| **Null safety** | `null` guards | Empty arrays `[||]` as default — no nulls |
| **Pipeline style** | LINQ method chains | `|>` pipe operator and `Seq` module |
| **Stack tuples** | `(Value, int)` | `struct (Value * int)` — stack-allocated, zero allocation |

### Advantages of the F# version

#### 1. Immutability by default
In F#, every binding is immutable unless you explicitly write `mutable`. This makes the data flow of the forward pass
and optimizer easier to reason about — mutation is visible and intentional, not accidental.

```fsharp
// Immutable by default
let x = rmsNorm x

// Mutation must be declared explicitly
let mutable loss = Value 0.0
loss <- loss + l
```

#### 2. Expressive pipeline syntax
The `|>` pipe operator lets you read data transformations left-to-right, matching how you think about them:

```fsharp
let docs =
    File.ReadAllLines "input.txt"
    |> Array.map    (fun l -> l.Trim())
    |> Array.filter (fun l -> not (String.IsNullOrEmpty l))
    |> (fun arr -> shuffle random (ResizeArray arr))
```

#### 3. Concise and noise-free syntax
F# requires no semicolons, fewer braces, and no `return` statements. The signal-to-noise ratio is higher, which helps
when studying an algorithm — you see the maths, not the ceremony.

#### 4. Strong type inference
F# infers types throughout, so you get full type safety without the verbosity of explicit annotations everywhere.

#### 5. Same performance
Both versions run on .NET and use the same `System.Numerics.Vector<double>` SIMD path inside `Value.Dot`. There is no
performance trade-off for choosing F#.

---

## Implementation notes (both versions)

Both implementations have **no external dependencies** beyond .NET itself. Everything — the autograd engine, the
transformer, the Adam optimizer, and the tokenizer — is implemented from scratch in a single file.

The [author](https://github.com/martinskuta) of the C# version deliberately optimized for raw CPU throughput. Several departures from the Python version were made on purpose:

- **SIMD vectorization** — `Value.Dot` uses `System.Numerics.Vector<double>` to process multiple elements per
  CPU instruction, giving a significant speedup over a scalar loop.
- **Iterative backward pass** — The original Python `backward()` is recursive and can hit Python's stack limit on
  long sequences. The C# version replaces recursion with an explicit `Stack<T>`, making it both faster and safe
  for deep graphs.
- **Zero-allocation hot paths** — `Value.Dot` pre-allocates the `children` and `localGrads` arrays once per node
  instead of creating intermediate `Value` objects for each multiply-and-add. This keeps GC pressure low during
  the training loop.
- **Backward loop unrolling** — The `Backward` method special-cases nodes with 1 or 2 children (which covers ~99%
  of the graph: Add, Mul, ReLU, Pow) to avoid loop setup overhead.

## Credits

Original [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) by Andrej Karpathy

C# translation by [@martinskuta](https://github.com/martinskuta)

F# translation by [@jonas1ara](https://github.com/jonas1ara)