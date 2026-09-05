// swift-tools-version: 6.3

import PackageDescription

#if os(macOS)
    let backendSwiftSettings: [SwiftSetting] = [
        .define("MLX_METAL_BACKEND")
    ]
#elseif os(Linux)
    let backendSwiftSettings: [SwiftSetting] =
        Context.environment["SPM_CUDA"] == "0"
        ? [.define("MLX_CPU_BACKEND")]
        : [.define("MLX_CUDA_BACKEND")]
#else
    let backendSwiftSettings: [SwiftSetting] = [
        .define("MLX_CPU_BACKEND")
    ]
#endif

#if os(Linux)
    // CUDA support currently lives on this exact post-0.31.6 MLX-Swift
    // revision. Keeping the selection in one conditional manifest lets this
    // package remain the canonical source tree for both CUDA and Metal.
    let mlxSwiftDependency: Package.Dependency = .package(
        url: "https://github.com/ml-explore/mlx-swift",
        revision: "2d2724006b62855c6c2a71df633baf4ee4ad8a0f"
    )
#else
    // The official 0.32 update branch synchronizes MLX Swift with MLX 0.32.2
    // and MLX-C. Pin the reviewed commit until it is published as a release.
    let mlxSwiftDependency: Package.Dependency = .package(
        url: "https://github.com/ml-explore/mlx-swift",
        revision: "72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"
    )
#endif

let package = Package(
    name: "Midnight",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "midnight", targets: ["Midnight"]),
        .executable(
            name: "model-runner-metal-quant-bench",
            targets: ["MetalQuantizationBenchmark"]
        ),
        .executable(
            name: "model-runner-laguna-quantize",
            targets: ["LagunaQuantizer"]
        ),
        .executable(
            name: "model-runner-scale-plan",
            targets: ["ScalePlanCLI"]
        ),
        .executable(
            name: "model-runner-q4-scale-search-audit",
            targets: ["Q4ScaleSearchAudit"]
        ),
        .executable(
            name: "model-runner-laguna-q4r8-rescore",
            targets: ["LagunaScaleSearchRescorerCLI"]
        ),
        .executable(
            name: "model-runner-quantize",
            targets: ["ModelQuantizer"]
        ),
        .executable(
            name: "model-runner-laguna-q4r8-verify",
            targets: ["LagunaQ4R8Verifier"]
        ),
        .executable(
            name: "model-runner-runtime-bench",
            targets: ["RuntimeBenchmark"]
        ),
        .executable(
            name: "model-runner-quality-bench",
            targets: ["ModelQualityBenchmark"]
        ),
        .executable(
            name: "model-runner-generation-bench",
            targets: ["ModelGenerationBenchmark"]
        ),
        .executable(
            name: "model-runner-mistral-activation-stats",
            targets: ["MistralActivationStats"]
        ),
        .executable(
            name: "model-runner-teacher-kl-bench",
            targets: ["ModelTeacherKLBenchmark"]
        ),
        .executable(
            name: "model-runner-mistral-awss-quantize",
            targets: ["MistralActivationScaleSearchRescorer"]
        ),
    ],
    dependencies: [
        mlxSwiftDependency,
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            revision: "14414441fa44f45eee35a61e9fa0bab577cf9734",
            traits: []
        ),
        .package(
            url: "https://github.com/huggingface/swift-huggingface",
            exact: "0.9.0"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            exact: "1.3.3"
        ),
        .package(
            url: "https://github.com/apple/swift-nio.git",
            exact: "2.101.3"
        ),
        .package(
            url: "https://github.com/apple/swift-argument-parser",
            exact: "1.8.2"
        ),
    ],
    targets: [
        .target(
            name: "ModelRunnerProtocol",
            swiftSettings: backendSwiftSettings
        ),
        .target(
            name: "ModelRunnerCore",
            dependencies: [
                "ModelRunnerProtocol",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "Midnight",
            dependencies: [
                "ModelRunnerCore",
                "ModelRunnerProtocol",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/ModelRunner",
            // The executable uses an @main AsyncParsableCommand and now has a
            // second source file for its POSIX signal relay.
            swiftSettings: [.unsafeFlags(["-parse-as-library"])]
        ),
        .executableTarget(
            name: "MetalQuantizationBenchmark",
            dependencies: [
                "ModelRunnerCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Benchmarks/MetalQuantization"
        ),
        .executableTarget(
            name: "LagunaQuantizer",
            dependencies: [
                "ModelRunnerCore",
                "ScalePlanMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "ScalePlanMLX"
        ),
        .executableTarget(
            name: "ScalePlanCLI",
            dependencies: [
                "ScalePlanMLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "Q4ScaleSearchAudit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "LagunaScaleSearchCore",
            dependencies: [
                "MistralActivationScaleSearchCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/LagunaScaleSearchRescorer"
        ),
        .executableTarget(
            name: "LagunaScaleSearchRescorerCLI",
            dependencies: [
                "LagunaScaleSearchCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "ModelQuantizer",
            dependencies: [
                "LagunaScaleSearchCore",
                "ModelRunnerCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "LagunaQ4R8Verifier",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "RuntimeBenchmark",
            dependencies: [
                "ModelRunnerCore",
                "ModelRunnerProtocol",
                "ModelQualityCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "ModelQualityCore"
        ),
        .executableTarget(
            name: "ModelGenerationBenchmark",
            dependencies: [
                "ModelQualityCore",
                "ModelRunnerCore",
                "ModelRunnerProtocol",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "ModelQualityBenchmark",
            dependencies: [
                "ModelQualityCore",
                "ModelRunnerCore",
                "ModelRunnerProtocol",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: backendSwiftSettings
        ),
        .executableTarget(
            name: "MistralActivationStats",
            dependencies: [
                "MistralActivationScaleSearchCore",
                "ModelQualityCore",
                "ModelRunnerCore",
                "ModelRunnerProtocol",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: backendSwiftSettings
        ),
        .executableTarget(
            name: "ModelTeacherKLBenchmark",
            dependencies: [
                "ModelQualityCore",
                "ModelRunnerCore",
                "ModelRunnerProtocol",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: backendSwiftSettings
        ),
        .target(
            name: "MistralActivationScaleSearchCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift")
            ]
        ),
        .executableTarget(
            name: "MistralActivationScaleSearchRescorer",
            dependencies: [
                "MistralActivationScaleSearchCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "ModelRunnerProtocolTests",
            dependencies: [
                "ModelRunnerCore",
                "ModelRunnerProtocol",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .testTarget(
            name: "ScalePlanMLXTests",
            dependencies: ["ScalePlanMLX"]
        ),
        .testTarget(
            name: "ModelQuantizerTests",
            dependencies: [
                "MistralActivationScaleSearchCore",
                "LagunaScaleSearchCore",
                "ModelRunnerCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .testTarget(
            name: "ModelQualityCoreTests",
            dependencies: ["ModelQualityCore"]
        ),
        .testTarget(
            name: "MistralActivationScaleSearchCoreTests",
            dependencies: [
                "MistralActivationScaleSearchCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
