// swift-tools-version: 6.3
import PackageDescription

#if os(macOS)
    let backendDependencies: [Package.Dependency] = [
        .package(url: "https://github.com/ml-explore/mlx-swift", revision: "72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"),
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm", revision: "14414441fa44f45eee35a61e9fa0bab577cf9734",
            traits: []),
        .package(url: "https://github.com/huggingface/swift-huggingface", exact: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
    ]
    let backendProducts: [Target.Dependency] = [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXVLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
        .product(name: "HuggingFace", package: "swift-huggingface"),
        .product(name: "Tokenizers", package: "swift-transformers"),
    ]
#else
    let backendDependencies: [Package.Dependency] = []
    let backendProducts: [Target.Dependency] = []
#endif

let package = Package(
    name: "MidnightVision",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "midnight-vision-worker", targets: ["MidnightVisionWorker"]),
        .library(name: "VisionProtocol", targets: ["VisionProtocol"]),
    ],
    dependencies: backendDependencies + [
        .package(path: "../../Shared/ModelFiles"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.101.3"),
    ],
    targets: [
        .target(name: "VisionProtocol"),
        .target(
            name: "VisionHTTP",
            dependencies: [
                "VisionProtocol", .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"), .product(name: "NIOPosix", package: "swift-nio"),
            ]),
        .executableTarget(
            name: "MidnightVisionWorker",
            dependencies: ["VisionProtocol", "VisionHTTP", .product(name: "ModelFiles", package: "ModelFiles")]
                + backendProducts, swiftSettings: [.unsafeFlags(["-parse-as-library"])]),
        .testTarget(
            name: "VisionProtocolTests",
            dependencies: [
                "VisionProtocol", "VisionHTTP", .product(name: "ModelFiles", package: "ModelFiles"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ]),
    ],
    swiftLanguageModes: [.v6]
)
