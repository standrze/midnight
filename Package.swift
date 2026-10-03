// swift-tools-version: 6.4

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
        .executable(name: "model-runner-prepare-corpus", targets: ["CorpusPreparation"]),
        // Sibling Training and Studio workers reuse this runtime.
        // Midnight has no dependency on those application/tool packages.
        .library(name: "ModelRunnerCore", targets: ["ModelRunnerCore"]),
        .library(name: "ModelRunnerProtocol", targets: ["ModelRunnerProtocol"]),
        .library(name: "ModelQualityCore", targets: ["ModelQualityCore"]),
        .executable(name: "midnight", targets: ["Midnight"]),
        .executable(
            name: "model-runner-runtime-bench",
            targets: ["RuntimeBenchmark"]
        ),
        .executable(
            name: "model-runner-paired-runtime-bench",
            targets: ["PairedRuntimeBenchmark"]
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
            name: "model-runner-teacher-kl-bench",
            targets: ["ModelTeacherKLBenchmark"]
        ),
        .executable(
            name: "model-runner-talkie-context-probe",
            targets: ["TalkieContextProbe"]
        ),
    ],
    dependencies: [
        .package(path: "Shared/ModelFiles"),
        .package(url: "https://github.com/standrze/loom.git", exact: "0.1.1"),
        .package(url: "https://github.com/standrze/weft.git", exact: "0.1.1"),
        mlxSwiftDependency,
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            revision: "14414441fa44f45eee35a61e9fa0bab577cf9734",
            traits: []
        ),
        .package(
            url: "https://github.com/Blaizzy/mlx-audio-swift",
            revision: "bf14ae0c26e4e85553dd989571cae29d70fa6735"
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
        .package(
            url: "https://github.com/apple/swift-crypto.git",
            exact: "4.5.1"
        ),
    ],
    targets: [
        .target(
            name: "CUDAMemory",
            cSettings: Context.environment["SPM_CUDA"] == "0"
                ? []
                : [
                    .define("MIDNIGHT_CUDA", .when(platforms: [.linux])),
                    .unsafeFlags(["-I/usr/local/cuda/include"], .when(platforms: [.linux])),
                ],
            linkerSettings: [.linkedLibrary("dl", .when(platforms: [.linux]))]
        ),
        .target(name: "CorpusPreparationCore", resources: [.copy("Resources/pinned-sources.json")]),
        .executableTarget(
            name: "CorpusPreparation",
            dependencies: [
                "CorpusPreparationCore", .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]),
        .testTarget(
            name: "CorpusPreparationTests", dependencies: ["CorpusPreparationCore"], resources: [.copy("Fixtures")]),
        .target(
            name: "ModelRunnerProtocol",
            swiftSettings: backendSwiftSettings
        ),
        .target(
            name: "ModelRunnerCore",
            dependencies: [
                "CUDAMemory",
                "ModelRunnerProtocol",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "Midnight",
            dependencies: [
                .product(name: "loom", package: "loom"),
                .product(name: "weft", package: "weft"),
                .product(name: "ModelFiles", package: "ModelFiles"),
                "ModelRunnerCore",
                "ModelRunnerProtocol",
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Crypto", package: "swift-crypto"),
            ],
            path: "Sources/ModelRunner",
            // The @main AsyncParsableCommand lives in main.swift, so parse it
            // as a library instead of treating that filename as top-level code.
            swiftSettings: [.unsafeFlags(["-parse-as-library"])]
        ),
        .executableTarget(
            name: "RuntimeBenchmark",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                "ModelRunnerCore",
                "ModelRunnerProtocol",
                "ModelQualityCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "PairedRuntimeBenchmark",
            dependencies: [
                "ModelQualityCore",
                "ModelRunnerCore",
                "ModelRunnerProtocol",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Crypto", package: "swift-crypto"),
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
        .executableTarget(
            name: "TalkieContextProbe",
            dependencies: [
                "CorpusPreparationCore",
                "ModelRunnerCore",
                "ModelRunnerProtocol",
                "ModelQualityCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: backendSwiftSettings + [.unsafeFlags(["-parse-as-library"])]
        ),
        .testTarget(
            name: "ModelRunnerProtocolTests",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                "ModelRunnerCore",
                "ModelRunnerProtocol",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            swiftSettings: backendSwiftSettings
        ),
        .testTarget(
            name: "MidnightTests",
            dependencies: [
                "Midnight",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOEmbedded", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
            ]
        ),
        .testTarget(
            name: "ModelQualityCoreTests",
            dependencies: ["ModelQualityCore"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
