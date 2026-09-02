import ArgumentParser
import Foundation
import LagunaScaleSearchCore
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import ModelRunnerCore

private struct IgnoredConfigurationValue: Decodable {}

private struct SourceDescriptor: Decodable {
  let modelType: String
  let architectures: [String]
  let quantization: IgnoredConfigurationValue?
  let quantizationConfig: IgnoredConfigurationValue?

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case architectures
    case quantization
    case quantizationConfig = "quantization_config"
  }

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    modelType = try container.decode(String.self, forKey: .modelType)
    architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
    quantization = try container.decodeIfPresent(
      IgnoredConfigurationValue.self, forKey: .quantization)
    quantizationConfig = try container.decodeIfPresent(
      IgnoredConfigurationValue.self, forKey: .quantizationConfig)
  }

  var isQuantized: Bool {
    quantization != nil || quantizationConfig != nil
  }

  var isLagunaDFlash: Bool {
    architectures.contains("DFlashLagunaForCausalLM")
  }
}

private struct QuantizerProvenance: Encodable {
  var format = 1
  var status: String
  var algorithm: String
  var createdAt: String
  var sourceModel: String
  var modelType: String
  var architectureProfile: String
  var bits = 4
  var groupSize = 64
  var q8Bits = 8
  var q4ScaleSearchModules: [String]
  var standardQ4Modules: [String]
  var q8Modules: [String]
  var mandatoryQ8Modules: [String]
  var skippedModules: [String]
  var outputShards: [String]

  enum CodingKeys: String, CodingKey {
    case format, status, algorithm, bits
    case createdAt = "created_at"
    case sourceModel = "source_model"
    case modelType = "model_type"
    case architectureProfile = "architecture_profile"
    case groupSize = "group_size"
    case q8Bits = "q8_bits"
    case q4ScaleSearchModules = "q4_scale_search_modules"
    case standardQ4Modules = "standard_q4_modules"
    case q8Modules = "q8_modules"
    case mandatoryQ8Modules = "mandatory_q8_modules"
    case skippedModules = "skipped_modules"
    case outputShards = "output_shards"
  }
}

private struct ArchitectureProfile {
  let name: String
  let mandatoryQ8Suffixes: [String]
  let requiresMandatoryQ8Match: Bool

  static let lagunaDFlash = Self(
    name: "laguna-dflash-q4r8",
    // DFlash's 10,240 -> 2,048 context projection is the sole bottleneck for
    // all target features. Its per-head attention gates are tiny and directly
    // modulate every draft layer. Protecting both costs little relative to the
    // 0.5B drafter while the larger Q/K/V/O and MLP matrices receive searched Q4.
    mandatoryQ8Suffixes: ["fc", ".self_attn.g_proj"],
    requiresMandatoryQ8Match: true
  )

  static func resolve(modelType: String) -> Self {
    switch modelType {
    case "laguna":
      return .init(
        name: "laguna-q4r8",
        mandatoryQ8Suffixes: [".mlp.gate.proj"],
        requiresMandatoryQ8Match: true
      )
    case "mixtral":
      return .init(
        name: "mixtral-moe-q4r8",
        mandatoryQ8Suffixes: [".block_sparse_moe.gate"],
        requiresMandatoryQ8Match: true
      )
    case "gpt_oss":
      return .init(
        name: "gpt-oss-moe-q4r8",
        mandatoryQ8Suffixes: [".mlp.router"],
        requiresMandatoryQ8Match: true
      )
    case "qwen3_moe", "qwen3_5_moe":
      return .init(
        name: "qwen-moe-q4r8",
        mandatoryQ8Suffixes: [".mlp.gate"],
        requiresMandatoryQ8Match: true
      )
    case "qwen3_next", "qwen3_5", "qwen3_5_text":
      // These model types can describe dense or routed variants. A discovered
      // MoE gate is protected, while a genuinely dense configuration remains Q4.
      return .init(
        name: "qwen-auto-q4r8",
        mandatoryQ8Suffixes: [".mlp.gate"],
        requiresMandatoryQ8Match: false
      )
    case "phimoe":
      return .init(
        name: "phi-moe-q4r8",
        mandatoryQ8Suffixes: [".block_sparse_moe.gate"],
        requiresMandatoryQ8Match: true
      )
    case "minimax":
      return .init(
        name: "minimax-moe-q4r8",
        mandatoryQ8Suffixes: [".block_sparse_moe.gate"],
        requiresMandatoryQ8Match: true
      )
    case "jamba":
      return .init(
        name: "jamba-moe-q4r8",
        mandatoryQ8Suffixes: [".block_sparse_moe.router"],
        requiresMandatoryQ8Match: true
      )
    default:
      return .init(
        name: "generic-mlx-q4",
        mandatoryQ8Suffixes: [],
        requiresMandatoryQ8Match: false
      )
    }
  }
}

private struct ConversionPlan {
  let modelType: String
  let profile: ArchitectureProfile
  let quantizablePaths: Set<String>
  let scaleSearchPaths: Set<String>
  let standardQ4Paths: Set<String>
  let q8Paths: Set<String>
  let mandatoryQ8Paths: Set<String>
  let skippedPaths: Set<String>
}

private enum QuantizerError: Error, LocalizedError {
  case invalidInput(String)
  case unsupportedModelType(String)
  case unsupportedDrafterArchitecture(String)
  case invalidPolicy(String)

  var errorDescription: String? {
    switch self {
    case .invalidInput(let message):
      "Invalid quantization input: \(message)"
    case .unsupportedModelType(let modelType):
      "Unsupported MLX Swift model_type '\(modelType)'. Add it to LLMTypeRegistry before quantizing it."
    case .unsupportedDrafterArchitecture(let architecture):
      "Unsupported MLX Swift drafter architecture '\(architecture)'. Add it to MTPDrafterTypeRegistry before quantizing it."
    case .invalidPolicy(let message):
      "Invalid quantization policy: \(message)"
    }
  }
}

@main
private struct ModelQuantizer: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "model-runner-quantize",
    abstract:
      "Quantize a registered MLX Swift safetensors LLM or DFlash drafter with Q4 ScaleSearch and architecture-aware Q8 modules.",
    discussion: """
      The source must be an unquantized safetensors checkpoint whose model_type is registered
      by MLX Swift LM. Mixtral, Laguna, GPT-OSS, and supported Qwen MoE models automatically
      keep their routing projections in standard affine Q8. Dense models such as Mistral and
      Llama use searched affine Q4 for eligible matrices.

      Poolside Laguna DFlash drafter checkpoints are detected by architecture and converted
      through the native drafter registry. Their context projection and per-head attention
      gates remain Q8; the larger draft projections use searched Q4.

      Laguna can additionally use --template to invoke the proven bounded-memory streaming
      converter while preserving its fused expert layout and exact Q8 router policy.
      """
  )

  @Argument(help: "Unquantized local safetensors LLM or DFlash drafter directory.")
  var source: String

  @Argument(help: "New destination model directory.")
  var destination: String

  @Option(
    name: .customLong("template"),
    help:
      "Laguna-only standard Q4R8 template; enables bounded shard conversion and preserves the proven fused layout."
  )
  var template: String?

  @Option(
    name: .customLong("q8-module"),
    help: "Module path or */? glob to keep in standard affine Q8; repeat as needed."
  )
  var q8ModulePatterns: [String] = []

  @Option(
    name: .customLong("skip-module"),
    help: "Module path or */? glob to leave unquantized; repeat as needed."
  )
  var skipModulePatterns: [String] = []

  @Option(
    name: .customLong("max-shard-gib"),
    help: "Maximum output safetensors shard size in GiB for generic conversion."
  )
  var maximumShardGiB = 5.0

  @Flag(
    name: .customLong("standard-q4"),
    help:
      "Use ordinary MLX affine Q4 group-64 calibration for eligible matrices as a matched ScaleSearch control."
  )
  var standardQ4 = false

  @Option(
    name: .customLong("expert-batch"),
    help: "Laguna --template expert batch size."
  )
  var expertBatch = 16

  @Flag(help: "Resolve and print the complete module policy without writing output.")
  var dryRun = false

  @Flag(help: "Run conversion using CPU/system memory instead of the default accelerator.")
  var cpu = false

  @Flag(help: "Replace an existing generic-conversion destination.")
  var overwrite = false

  mutating func validate() throws {
    guard maximumShardGiB.isFinite, maximumShardGiB > 0 else {
      throw ValidationError("--max-shard-gib must be a finite positive number.")
    }
    let bytes = maximumShardGiB * 1_024 * 1_024 * 1_024
    guard bytes <= Double(Int64.max) else {
      throw ValidationError("--max-shard-gib is too large.")
    }
    guard (1 ... 256).contains(expertBatch) else {
      throw ValidationError("--expert-batch must be in 1...256.")
    }
    let blankPatterns = (q8ModulePatterns + skipModulePatterns).filter {
      $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
    guard blankPatterns.isEmpty else {
      throw ValidationError("--q8-module and --skip-module patterns must be nonblank.")
    }
    if template != nil {
      guard q8ModulePatterns.isEmpty, skipModulePatterns.isEmpty else {
        throw ValidationError(
          "--template uses Laguna's exact Q4R8 layout; custom Q8 and skip patterns are not accepted."
        )
      }
      guard !overwrite else {
        throw ValidationError(
          "--overwrite is not supported with --template; choose a new destination."
        )
      }
      guard !standardQ4 else {
        throw ValidationError(
          "--standard-q4 is a generic-conversion control; Laguna --template retains its exact searched-Q4 layout."
        )
      }
    }
  }

  mutating func run() async throws {
    #if os(Linux)
      defer { clearStreams() }
    #endif

    if let template {
      try runLagunaTemplate(template)
      return
    }

    if dryRun || cpu {
      try await Device.withDefaultDevice(.cpu) {
        try await runGenericConversion()
      }
    } else {
      try await runGenericConversion()
    }
  }

  private func runLagunaTemplate(_ template: String) throws {
    let sourceURL = URL(fileURLWithPath: source).standardizedFileURL
    let descriptor = try readDescriptor(sourceURL: sourceURL)
    guard descriptor.modelType == "laguna" else {
      throw QuantizerError.invalidPolicy(
        "--template is Laguna-specific, but source model_type is '\(descriptor.modelType)'."
      )
    }
    guard !descriptor.isLagunaDFlash else {
      throw QuantizerError.invalidPolicy(
        "--template applies to the Laguna target, not its separate DFlash drafter checkpoint."
      )
    }
    print("Profile: laguna-q4r8-template-streaming")
    print("Policy: searched affine Q4 group-64; standard affine Q8 routers preserved")
    try LagunaScaleSearchRescorer.rescore(
      source: source,
      template: template,
      destination: destination,
      expertBatch: expertBatch,
      preflightOnly: dryRun,
      cpu: cpu || dryRun
    )
  }

  private func runGenericConversion() async throws {
    let sourceURL = URL(fileURLWithPath: source).standardizedFileURL
    let destinationURL = URL(fileURLWithPath: destination).standardizedFileURL
    let descriptor = try readDescriptor(sourceURL: sourceURL)
    guard !descriptor.isQuantized else {
      throw QuantizerError.invalidInput(
        "source config already declares quantization; requantization is intentionally unsupported"
      )
    }
    guard sourceURL != destinationURL else {
      throw QuantizerError.invalidInput("source and destination must be distinct")
    }

    let configurationData = try Data(
      contentsOf: sourceURL.appendingPathComponent("config.json"))
    if descriptor.isLagunaDFlash {
      try await runDFlashConversion(
        sourceURL: sourceURL,
        destinationURL: destinationURL,
        descriptor: descriptor,
        configurationData: configurationData
      )
      return
    }

    await LagunaModelRegistration.register()
    guard await LLMTypeRegistry.shared.contains(descriptor.modelType) else {
      throw QuantizerError.unsupportedModelType(descriptor.modelType)
    }
    let model = try await LLMTypeRegistry.shared.createModel(
      configuration: configurationData,
      modelType: descriptor.modelType
    )
    let plan = try makePlan(model: model, modelType: descriptor.modelType)
    printPlan(plan, device: dryRun ? "CPU policy inspection" : String(describing: Device.defaultDevice()))
    if dryRun { return }

    let q8 = ModelConversionQuantization(
      bits: 8,
      groupSize: 64,
      mode: .affine,
      calibration: .standard
    )
    let q8Paths = plan.q8Paths
    let skippedPaths = plan.skippedPaths
    let options = ModelConversionOptions(
      bits: 4,
      groupSize: 64,
      mode: .affine,
      calibration: standardQ4 ? .standard : .q4AffineScaleSearch,
      maxShardSize: Int64(maximumShardGiB * 1_024 * 1_024 * 1_024),
      overwriteExistingOutput: overwrite,
      quantizationPredicate: { path, _ in
        if skippedPaths.contains(path) {
          return .skip
        }
        if q8Paths.contains(path) {
          return .quantize(q8)
        }
        return .quantize()
      }
    )

    let result = try await LLMModelFactory.shared.convert(
      from: sourceURL,
      to: destinationURL,
      options: options,
      progressHandler: { progress in
        let detail = progress.message.map { ": \($0)" } ?? ""
        print("[\(progress.stage.rawValue)]\(detail)")
      }
    )
    try writeProvenance(plan: plan, result: result, sourceURL: sourceURL)
    print(
      "Created \(result.outputDirectory.path) with \(result.weightsURLs.count) weight shard(s)."
    )
  }

  private func runDFlashConversion(
    sourceURL: URL,
    destinationURL: URL,
    descriptor: SourceDescriptor,
    configurationData: Data
  ) async throws {
    await LagunaDFlashRegistration.register()
    let model: any MTPDrafterModel
    do {
      model = try await MTPDrafterTypeRegistry.shared.createModel(
        configuration: configurationData,
        modelType: descriptor.modelType
      )
    } catch {
      throw QuantizerError.unsupportedDrafterArchitecture(
        "\(descriptor.modelType) / DFlashLagunaForCausalLM")
    }

    let plan = try makePlan(
      model: model,
      modelType: descriptor.modelType,
      profile: .lagunaDFlash
    )
    printPlan(
      plan,
      device: dryRun ? "CPU policy inspection" : String(describing: Device.defaultDevice())
    )
    print("DFlash note: compare acceptance and end-to-end throughput against the BF16 drafter.")
    if dryRun { return }

    let q8 = ModelConversionQuantization(
      bits: 8,
      groupSize: 64,
      mode: .affine,
      calibration: .standard
    )
    let q8Paths = plan.q8Paths
    let skippedPaths = plan.skippedPaths
    let options = ModelConversionOptions(
      bits: 4,
      groupSize: 64,
      mode: .affine,
      calibration: standardQ4 ? .standard : .q4AffineScaleSearch,
      maxShardSize: Int64(maximumShardGiB * 1_024 * 1_024 * 1_024),
      overwriteExistingOutput: overwrite,
      quantizationPredicate: { path, _ in
        if skippedPaths.contains(path) { return .skip }
        if q8Paths.contains(path) { return .quantize(q8) }
        return .quantize()
      }
    )
    let result = try MLXLMCommon.convert(
      modelDirectory: sourceURL,
      model: model,
      to: destinationURL,
      options: options,
      progressHandler: { progress in
        let detail = progress.message.map { ": \($0)" } ?? ""
        print("[\(progress.stage.rawValue)]\(detail)")
      }
    )
    try writeProvenance(plan: plan, result: result, sourceURL: sourceURL)
    print(
      "Created \(result.outputDirectory.path) with \(result.weightsURLs.count) DFlash weight shard(s)."
    )
  }

  private func readDescriptor(sourceURL: URL) throws -> SourceDescriptor {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: sourceURL.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw QuantizerError.invalidInput(
        "source directory does not exist: \(sourceURL.path)")
    }
    let configURL = sourceURL.appendingPathComponent("config.json")
    do {
      return try JSONDecoder.json5().decode(
        SourceDescriptor.self,
        from: Data(contentsOf: configURL)
      )
    } catch {
      throw QuantizerError.invalidInput(
        "cannot decode \(configURL.path): \(error.localizedDescription)"
      )
    }
  }

  private func makePlan(
    model: any BaseLanguageModel,
    modelType: String,
    profile requestedProfile: ArchitectureProfile? = nil
  ) throws -> ConversionPlan {
    let modules = model.leafModules().flattened()
    let quantizable = modules.filter { _, module in
      module is Quantizable && !(module is Quantized)
    }
    let quantizablePaths = Set(quantizable.map(\.0))
    guard !quantizablePaths.isEmpty else {
      throw QuantizerError.invalidPolicy(
        "model exposes no unquantized Quantizable modules")
    }

    let profile = requestedProfile ?? ArchitectureProfile.resolve(modelType: modelType)
    let mandatoryQ8Paths = Set(
      quantizablePaths.filter { path in
        profile.mandatoryQ8Suffixes.contains { path.hasSuffix($0) }
      }
    )
    if profile.requiresMandatoryQ8Match, mandatoryQ8Paths.isEmpty {
      throw QuantizerError.invalidPolicy(
        "profile \(profile.name) found no expected mandatory Q8 module matching "
          + profile.mandatoryQ8Suffixes.joined(separator: ", ")
      )
    }

    let requestedQ8 = try resolvePatterns(
      q8ModulePatterns,
      paths: quantizablePaths,
      optionName: "--q8-module"
    )
    let requestedSkips = try resolvePatterns(
      skipModulePatterns,
      paths: quantizablePaths,
      optionName: "--skip-module"
    )
    let q8Paths = mandatoryQ8Paths.union(requestedQ8)
    let skippedMandatory = requestedSkips.intersection(mandatoryQ8Paths)
    guard skippedMandatory.isEmpty else {
      throw QuantizerError.invalidPolicy(
        "mandatory Q8 module(s) cannot be skipped: "
          + skippedMandatory.sorted().joined(separator: ", ")
      )
    }
    let conflicts = requestedSkips.intersection(q8Paths)
    guard conflicts.isEmpty else {
      throw QuantizerError.invalidPolicy(
        "modules cannot be both Q8 and skipped: "
          + conflicts.sorted().joined(separator: ", ")
      )
    }

    let selectedQ4Paths = quantizablePaths.subtracting(q8Paths).subtracting(requestedSkips)
    var scaleSearchPaths = Set<String>()
    var standardQ4Paths = Set<String>()
    var incompatiblePaths = [String]()
    for (path, module) in quantizable where selectedQ4Paths.contains(path) {
      if let width = groupWidth(module), width % 64 != 0 {
        incompatiblePaths.append("\(path) (input width \(width))")
      } else if module is Linear || module is SwitchLinear {
        scaleSearchPaths.insert(path)
      } else {
        // Embeddings and custom Quantizable modules retain standard MLX Q4.
        // They share the same load/runtime format but do not accept searched arrays.
        standardQ4Paths.insert(path)
      }
    }
    for (path, module) in quantizable where q8Paths.contains(path) {
      if let width = groupWidth(module), width % 64 != 0 {
        incompatiblePaths.append("\(path) (Q8 input width \(width))")
      }
    }
    guard incompatiblePaths.isEmpty else {
      throw QuantizerError.invalidPolicy(
        "group-64 cannot represent these modules; explicitly --skip-module them or use a compatible architecture: "
          + incompatiblePaths.sorted().joined(separator: ", ")
      )
    }
    guard !scaleSearchPaths.isEmpty else {
      throw QuantizerError.invalidPolicy(
        "model exposes no group-64 Linear or SwitchLinear matrices for "
          + (standardQ4 ? "standard Q4" : "ScaleSearch")
      )
    }

    return ConversionPlan(
      modelType: modelType,
      profile: profile,
      quantizablePaths: quantizablePaths,
      scaleSearchPaths: scaleSearchPaths,
      standardQ4Paths: standardQ4Paths,
      q8Paths: q8Paths,
      mandatoryQ8Paths: mandatoryQ8Paths,
      skippedPaths: requestedSkips
    )
  }

  private func groupWidth(_ module: Module) -> Int? {
    if let linear = module as? Linear { return linear.weight.dim(-1) }
    if module is SwitchLinear {
      return module.parameters().flattened().first { $0.0 == "weight" }?.1.dim(-1)
    }
    if let embedding = module as? Embedding { return embedding.weight.dim(-1) }
    return nil
  }

  private func resolvePatterns(
    _ patterns: [String],
    paths: Set<String>,
    optionName: String
  ) throws -> Set<String> {
    var resolved = Set<String>()
    for rawPattern in patterns {
      var pattern = rawPattern.trimmingCharacters(in: .whitespacesAndNewlines)
      if pattern.hasSuffix(".weight") {
        pattern.removeLast(".weight".count)
      }
      let matches = paths.filter { Self.glob(pattern, matches: $0) }
      guard !matches.isEmpty else {
        throw QuantizerError.invalidPolicy(
          "\(optionName) pattern '\(rawPattern)' matched no quantizable module"
        )
      }
      resolved.formUnion(matches)
    }
    return resolved
  }

  private static func glob(_ pattern: String, matches text: String) -> Bool {
    let pattern = Array(pattern)
    let text = Array(text)
    var patternIndex = 0
    var textIndex = 0
    var starIndex: Int?
    var retryTextIndex = 0

    while textIndex < text.count {
      if patternIndex < pattern.count,
        pattern[patternIndex] == "?" || pattern[patternIndex] == text[textIndex]
      {
        patternIndex += 1
        textIndex += 1
      } else if patternIndex < pattern.count, pattern[patternIndex] == "*" {
        starIndex = patternIndex
        patternIndex += 1
        retryTextIndex = textIndex
      } else if let starIndex {
        patternIndex = starIndex + 1
        retryTextIndex += 1
        textIndex = retryTextIndex
      } else {
        return false
      }
    }
    while patternIndex < pattern.count, pattern[patternIndex] == "*" {
      patternIndex += 1
    }
    return patternIndex == pattern.count
  }

  private func printPlan(_ plan: ConversionPlan, device: String) {
    print("Model type: \(plan.modelType)")
    print("Profile: \(plan.profile.name)")
    print("Device: \(device)")
    if standardQ4 {
      print("Policy: standard affine group-64 Q4 with standard affine Q8 overrides")
    } else {
      print("Policy: affine group-64 Q4 ScaleSearch with standard affine Q8 overrides")
    }
    print("Quantizable modules: \(plan.quantizablePaths.count)")
    if standardQ4 {
      print("  standard Q4 Linear/SwitchLinear: \(plan.scaleSearchPaths.count)")
    } else {
      print("  Q4 ScaleSearch Linear/SwitchLinear: \(plan.scaleSearchPaths.count)")
    }
    print("  standard Q4 embedding/custom modules: \(plan.standardQ4Paths.count)")
    print("  standard Q8 modules: \(plan.q8Paths.count)")
    print("  skipped modules: \(plan.skippedPaths.count)")
    if !plan.q8Paths.isEmpty {
      print("Q8 module paths:")
      for path in plan.q8Paths.sorted() {
        let origin = plan.mandatoryQ8Paths.contains(path) ? "mandatory" : "requested"
        print("  \(path) [\(origin)]")
      }
    }
    if !plan.skippedPaths.isEmpty {
      print("Skipped module paths:")
      for path in plan.skippedPaths.sorted() {
        print("  \(path)")
      }
    }
  }

  private func writeProvenance(
    plan: ConversionPlan,
    result: ModelConversionResult,
    sourceURL: URL
  ) throws {
    let isDFlash = plan.profile.name == ArchitectureProfile.lagunaDFlash.name
    let standardQ4Paths =
      standardQ4
      ? plan.scaleSearchPaths.union(plan.standardQ4Paths).sorted()
      : plan.standardQ4Paths.sorted()
    let provenance = QuantizerProvenance(
      status:
        standardQ4
        ? (isDFlash
          ? "experimental_unbenchmarked_dflash_standard_q4_control"
          : "experimental_unbenchmarked_standard_q4_control")
        : (isDFlash
          ? "experimental_unbenchmarked_dflash_candidate"
          : "experimental_measured_calibration"),
      algorithm: standardQ4 ? "mlx_affine_q4_standard" : "q4r8_affine_scale_search_ls2",
      createdAt: ISO8601DateFormatter().string(from: Date()),
      sourceModel: sourceURL.path,
      modelType: plan.modelType,
      architectureProfile: plan.profile.name,
      q4ScaleSearchModules: standardQ4 ? [] : plan.scaleSearchPaths.sorted(),
      standardQ4Modules: standardQ4Paths,
      q8Modules: plan.q8Paths.sorted(),
      mandatoryQ8Modules: plan.mandatoryQ8Paths.sorted(),
      skippedModules: plan.skippedPaths.sorted(),
      outputShards: result.weightsURLs.map(\.lastPathComponent).sorted()
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(provenance).write(
      to: result.outputDirectory.appendingPathComponent(
        standardQ4
          ? "standard-q4-quantization.json"
          : "scale-search-quantization.json"),
      options: .atomic
    )
  }
}
