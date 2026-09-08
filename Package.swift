// swift-tools-version: 6.3

import PackageDescription

// The optional native audio backend currently requires AVFoundation.
#if os(macOS)
let chatterboxPackages: [Package.Dependency] = [
    .package(url: "https://github.com/Blaizzy/mlx-audio-swift", revision: "bf14ae0c26e4e85553dd989571cae29d70fa6735")
]
let chatterboxProducts: [Target.Dependency] = [
    .product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
    .product(name: "MLXAudioCore", package: "mlx-audio-swift")
]
#else
let chatterboxPackages: [Package.Dependency] = []
let chatterboxProducts: [Target.Dependency] = []
#endif

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
        // Sibling Training, Studio workers, and Quantization reuse this runtime.
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
    ],
    dependencies: chatterboxPackages + [
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
        .target(name: "CorpusPreparationCore", resources: [.copy("Resources/pinned-sources.json")]),
        .executableTarget(name: "CorpusPreparation", dependencies: [
            "CorpusPreparationCore", .product(name: "ArgumentParser", package: "swift-argument-parser")
        ]),
        .testTarget(name: "CorpusPreparationTests", dependencies: ["CorpusPreparationCore"], resources: [.copy("Fixtures")]),
        .target(
            name: "ModelRunnerProtocol",
            swiftSettings: backendSwiftSettings
        ),
        .target(
            name: "ModelRunnerCore",
            dependencies: chatterboxProducts + [
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
                .product(name: "HuggingFace", package: "swift-huggingface"),
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
            name: "MidnightTests",
            dependencies: [
                "Midnight",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOEmbedded", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ]
        ),
        .testTarget(
            name: "ModelQualityCoreTests",
            dependencies: ["ModelQualityCore"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
