# Third-party notices

Midnight is distributed under the Apache License 2.0. It depends on and carries
patches for third-party projects that remain under their original licenses.
Nothing in Midnight's license replaces those terms.

## MLX projects

- [mlx-swift](https://github.com/ml-explore/mlx-swift) — MIT License,
  copyright (c) 2023 ml-explore.
- [MLX](https://github.com/ml-explore/mlx) — MIT License,
  copyright © 2023 Apple Inc.
- [MLX-C](https://github.com/ml-explore/mlx-c) — MIT License,
  copyright (c) 2023 ml-explore.
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) — MIT License,
  copyright (c) 2024 ml-explore.

The following MIT terms apply to those projects and to portions represented in
the corresponding files under `Patches/`:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Apache-licensed dependencies

The following direct dependencies are licensed under the Apache License 2.0:

- [swift-huggingface](https://github.com/huggingface/swift-huggingface)
- [swift-transformers](https://github.com/huggingface/swift-transformers)
- [SwiftNIO](https://github.com/apple/swift-nio)
- [Swift Argument Parser](https://github.com/apple/swift-argument-parser)

The Apache License 2.0 text is included in [`LICENSE`](LICENSE). SwiftNIO also
publishes an upstream [`NOTICE.txt`](https://github.com/apple/swift-nio/blob/2.101.3/NOTICE.txt)
covering components incorporated by SwiftNIO. This repository does not publish
a prebuilt binary; preserve the full notice from the pinned dependency when
redistributing SwiftNIO or a binary distribution that requires it.

Swift Package Manager resolves the precise dependency revisions recorded in
`Package.swift` and `Package.resolved`. Model checkpoints are not distributed
with Midnight and retain their own licenses.
