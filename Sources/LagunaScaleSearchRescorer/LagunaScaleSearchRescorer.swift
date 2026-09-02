import ArgumentParser
import Foundation
import MLX
import MLXLMCommon

private struct SafetensorsIndex: Decodable {
  var weightMap: [String: String]

  enum CodingKeys: String, CodingKey {
    case weightMap = "weight_map"
  }
}

private struct SourceConfiguration: Decodable {
  var numberOfExperts: Int

  enum CodingKeys: String, CodingKey {
    case numberOfExperts = "num_experts"
  }
}

private struct RescoreProvenance: Encodable {
  var format = 1
  var status = "experimental_measured_candidate"
  var algorithm = "q4r8_affine_scale_search_ls2"
  var createdAt: String
  var sourceModel: String
  var templateModel: String
  var bits = 4
  var groupSize = 64
  var searchFactors: [Double] = [
    0.75, 0.8125, 0.875, 0.9375, 1,
    1.0625, 1.125, 1.1875, 1.25,
  ]
  var biasRefinementIterations = 1
  var jointAffineRefinementIterations = 2
  var q4ModulesRescored: Int
  var q8ModulesPreserved: Int
  var standardQ4EmbeddingsPreserved: Int
  var expertBatchSize: Int

  enum CodingKeys: String, CodingKey {
    case format, status, algorithm, bits
    case createdAt = "created_at"
    case sourceModel = "source_model"
    case templateModel = "template_model"
    case groupSize = "group_size"
    case searchFactors = "search_factors"
    case biasRefinementIterations = "bias_refinement_iterations"
    case jointAffineRefinementIterations = "joint_affine_refinement_iterations"
    case q4ModulesRescored = "q4_modules_rescored"
    case q8ModulesPreserved = "q8_modules_preserved"
    case standardQ4EmbeddingsPreserved = "standard_q4_embeddings_preserved"
    case expertBatchSize = "expert_batch_size"
  }
}

private struct QuantizedArrays {
  var weight: MLXArray
  var scales: MLXArray
  var biases: MLXArray
}

private enum RescoreError: Error, LocalizedError {
  case invalidInput(String)
  case incompatibleTemplate(String)
  case missingTensor(String)

  var errorDescription: String? {
    switch self {
    case .invalidInput(let message): "Invalid rescore input: \(message)"
    case .incompatibleTemplate(let message): "Incompatible Q4R8 template: \(message)"
    case .missingTensor(let key): "Missing source tensor: \(key)"
    }
  }
}

public struct LagunaScaleSearchRescorer: ParsableCommand {
  public static let configuration = CommandConfiguration(
    commandName: "model-runner-laguna-q4r8-rescore",
    abstract:
      "Recompute Laguna Q4 tensors with affine ScaleSearch while streaming through a standard Q4R8 layout."
  )

  @Argument(help: "Original unquantized Laguna safetensors directory.")
  var source: String

  @Argument(help: "Read-only standard MLX Q4R8 checkpoint used for layout and preserved Q8 arrays.")
  var template: String

  @Argument(help: "New destination directory; it must not already exist.")
  var destination: String

  @Option(
    name: .customLong("expert-batch"),
    help: "Number of routed experts to quantize in each bounded GPU batch."
  )
  var expertBatch = 16

  @Flag(help: "Validate source/template identity and mappings without writing a destination.")
  var preflightOnly = false

  @Flag(help: "Run searched quantization on CPU rather than the default GPU.")
  var cpu = false

  public init() {}

  public static func rescore(
    source: String,
    template: String,
    destination: String,
    expertBatch: Int = 16,
    preflightOnly: Bool = false,
    cpu: Bool = false
  ) throws {
    var command = Self()
    command.source = source
    command.template = template
    command.destination = destination
    command.expertBatch = expertBatch
    command.preflightOnly = preflightOnly
    command.cpu = cpu
    try command.validate()
    try command.run()
  }

  public mutating func validate() throws {
    guard expertBatch >= 1, expertBatch <= 256 else {
      throw ValidationError("--expert-batch must be in 1...256.")
    }
  }

  public mutating func run() throws {
    #if os(Linux)
      defer { clearStreams() }
    #endif

    if cpu {
      try Device.withDefaultDevice(.cpu) { try execute() }
    } else {
      try execute()
    }
  }

  private func execute() throws {
    Memory.cacheLimit = 512 * 1_024 * 1_024

    let sourceURL = URL(fileURLWithPath: source).standardizedFileURL
    let templateURL = URL(fileURLWithPath: template).standardizedFileURL
    let destinationURL = URL(fileURLWithPath: destination).standardizedFileURL
    try validateDirectory(sourceURL, label: "source")
    try validateDirectory(templateURL, label: "template")
    guard sourceURL != templateURL, sourceURL != destinationURL, templateURL != destinationURL else {
      throw RescoreError.invalidInput("source, template, and destination must be distinct")
    }
    guard !FileManager.default.fileExists(atPath: destinationURL.path) else {
      throw RescoreError.invalidInput("destination already exists: \(destinationURL.path)")
    }

    let sourceIndexURL = sourceURL.appendingPathComponent("model.safetensors.index.json")
    let templateIndexURL = templateURL.appendingPathComponent("model.safetensors.index.json")
    let sourceIndex = try decodeIndex(at: sourceIndexURL)
    let templateIndex = try decodeIndex(at: templateIndexURL)
    let sourceConfiguration = try JSONDecoder().decode(
      SourceConfiguration.self,
      from: Data(contentsOf: sourceURL.appendingPathComponent("config.json"))
    )
    guard sourceConfiguration.numberOfExperts > 0 else {
      throw RescoreError.invalidInput("source num_experts must be positive")
    }

    let quantizedModules = templateIndex.weightMap.keys.compactMap { key -> String? in
      guard key.hasSuffix(".weight") else { return nil }
      let module = String(key.dropLast(".weight".count))
      guard templateIndex.weightMap["\(module).scales"] != nil,
        templateIndex.weightMap["\(module).biases"] != nil
      else { return nil }
      return module
    }.sorted()
    let q8Modules = quantizedModules.filter(Self.isQ8Router)
    let embeddingModules = quantizedModules.filter(Self.isStandardEmbedding)
    let q4Modules = quantizedModules.filter {
      !Self.isQ8Router($0) && !Self.isStandardEmbedding($0)
    }
    guard !q4Modules.isEmpty else {
      throw RescoreError.incompatibleTemplate("no affine Q4 modules were found")
    }
    try validateSourceMappings(
      q4Modules,
      sourceWeightMap: sourceIndex.weightMap,
      numberOfExperts: sourceConfiguration.numberOfExperts
    )

    print(
      "Laguna Q4R8 LS2 rescore preflight: \(q4Modules.count) Q4 modules, "
        + "\(q8Modules.count) Q8 routers preserved, "
        + "\(embeddingModules.count) standard Q4 embedding(s) preserved"
    )
    print(
      "source tensors=\(sourceIndex.weightMap.count) template tensors=\(templateIndex.weightMap.count) "
        + "experts=\(sourceConfiguration.numberOfExperts) device=\(Device.defaultDevice())"
    )

    let sourceArrays = try loadSourceArrays(sourceURL: sourceURL, index: sourceIndex)
    try verifyTemplateIdentity(
      sourceArrays: sourceArrays,
      templateURL: templateURL,
      templateIndex: templateIndex
    )
    print("Template identity checks passed for direct Q4, routed Q4, fused gate/up Q4, and router Q8.")
    if preflightOnly { return }

    try FileManager.default.createDirectory(
      at: destinationURL, withIntermediateDirectories: false)
    try copySidecars(from: templateURL, to: destinationURL)

    let shardNames = Set(templateIndex.weightMap.values).sorted()
    let shardOrder = Dictionary(
      uniqueKeysWithValues: shardNames.enumerated().map { ($0.element, $0.offset) }
    )
    var modulesByShard = [String: [String]]()
    for module in q4Modules {
      let moduleShards = try ["weight", "scales", "biases"].map { suffix in
        guard let shard = templateIndex.weightMap["\(module).\(suffix)"] else {
          throw RescoreError.incompatibleTemplate("missing \(module).\(suffix)")
        }
        return shard
      }
      guard let firstShard = moduleShards.min(by: {
        shardOrder[$0, default: .max] < shardOrder[$1, default: .max]
      }) else {
        throw RescoreError.incompatibleTemplate("missing shard placement for \(module)")
      }
      modulesByShard[firstShard, default: []].append(module)
    }
    var pendingReplacements = [String: MLXArray]()
    var completedModules = 0
    for (shardOffset, shardName) in shardNames.enumerated() {
      let sourceShardURL = templateURL.appendingPathComponent(shardName)
      var arrays = try loadArrays(url: sourceShardURL, stream: .cpu)
      let modules = (modulesByShard[shardName] ?? []).sorted()
      print("[shard \(shardOffset + 1)/\(shardNames.count)] \(shardName): \(modules.count) Q4 module(s)")

      for key in pendingReplacements.keys.sorted() where arrays[key] != nil {
        let value = pendingReplacements.removeValue(forKey: key)!
        try install(value, key: key, arrays: &arrays)
      }

      for module in modules {
        let replacement: QuantizedArrays
        if let routed = Self.routedProjection(module) {
          replacement = try quantizeRoutedProjection(
            routed,
            sourceArrays: sourceArrays,
            numberOfExperts: sourceConfiguration.numberOfExperts
          )
        } else {
          let sourceKey = try Self.directSourceWeightKey(for: module)
          guard let sourceWeight = sourceArrays[sourceKey] else {
            throw RescoreError.missingTensor(sourceKey)
          }
          replacement = searched(sourceWeight)
        }

        for (key, value) in replacementEntries(replacement, module: module) {
          if arrays[key] != nil {
            try install(value, key: key, arrays: &arrays)
          } else {
            guard templateIndex.weightMap[key] != nil,
              pendingReplacements.updateValue(value, forKey: key) == nil
            else {
              throw RescoreError.incompatibleTemplate(
                "invalid or duplicate deferred replacement \(key)"
              )
            }
          }
        }
        completedModules += 1
        print("  [\(completedModules)/\(q4Modules.count)] \(module)")
        Memory.clearCache()
      }

      Stream.defaultStream(Device.defaultDevice()).synchronize()
      let temporaryURL = destinationURL.appendingPathComponent(".\(shardName).partial.safetensors")
      try save(arrays: arrays, metadata: ["format": "mlx"], url: temporaryURL)
      let outputURL = destinationURL.appendingPathComponent(shardName)
      try FileManager.default.moveItem(at: temporaryURL, to: outputURL)
      arrays.removeAll(keepingCapacity: false)
      Memory.clearCache()
    }
    guard pendingReplacements.isEmpty, completedModules == q4Modules.count else {
      throw RescoreError.incompatibleTemplate(
        "stream ended with \(pendingReplacements.count) deferred tensor(s) and "
          + "\(completedModules)/\(q4Modules.count) Q4 modules"
      )
    }

    try FileManager.default.copyItem(
      at: templateIndexURL,
      to: destinationURL.appendingPathComponent("model.safetensors.index.json")
    )
    let provenance = RescoreProvenance(
      createdAt: ISO8601DateFormatter().string(from: Date()),
      sourceModel: sourceURL.path,
      templateModel: templateURL.path,
      q4ModulesRescored: q4Modules.count,
      q8ModulesPreserved: q8Modules.count,
      standardQ4EmbeddingsPreserved: embeddingModules.count,
      expertBatchSize: expertBatch
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(provenance).write(
      to: destinationURL.appendingPathComponent("q4r8-scale-search.json"),
      options: .atomic
    )
    print("Created \(destinationURL.path) from \(shardNames.count) streamed shard(s).")
  }

  private func loadSourceArrays(
    sourceURL: URL,
    index: SafetensorsIndex
  ) throws -> [String: MLXArray] {
    let shardNames = Set(index.weightMap.values).sorted()
    var result = [String: MLXArray]()
    result.reserveCapacity(index.weightMap.count)
    for (offset, shardName) in shardNames.enumerated() {
      let shardURL = sourceURL.appendingPathComponent(shardName)
      let arrays = try loadArrays(url: shardURL, stream: .cpu)
      for (key, value) in arrays {
        guard result.updateValue(value, forKey: key) == nil else {
          throw RescoreError.invalidInput("duplicate source tensor \(key)")
        }
      }
      print("[source \(offset + 1)/\(shardNames.count)] indexed \(shardName)")
    }
    guard result.count == index.weightMap.count else {
      throw RescoreError.invalidInput(
        "source index names \(index.weightMap.count) tensors but shards exposed \(result.count)")
    }
    return result
  }

  private func searched(_ source: MLXArray) -> QuantizedArrays {
    let result = q4AffineScaleSearchQuantized(source)
    MLX.eval(result.weight, result.scales, result.biases)
    Stream.defaultStream(Device.defaultDevice()).synchronize()
    return .init(weight: result.weight, scales: result.scales, biases: result.biases)
  }

  private func quantizeRoutedProjection(
    _ routed: (layer: Int, projection: String),
    sourceArrays: [String: MLXArray],
    numberOfExperts: Int
  ) throws -> QuantizedArrays {
    let prefix = "model.layers.\(routed.layer).mlp.experts"
    var weightParts = [MLXArray]()
    var scaleParts = [MLXArray]()
    var biasParts = [MLXArray]()
    weightParts.reserveCapacity((numberOfExperts + expertBatch - 1) / expertBatch)
    scaleParts.reserveCapacity(weightParts.capacity)
    biasParts.reserveCapacity(weightParts.capacity)

    for start in stride(from: 0, to: numberOfExperts, by: expertBatch) {
      let end = min(start + expertBatch, numberOfExperts)
      var batch = [MLXArray]()
      batch.reserveCapacity(end - start)
      for expert in start ..< end {
        if routed.projection == "gate_up_proj" {
          let gateKey = "\(prefix).\(expert).gate_proj.weight"
          let upKey = "\(prefix).\(expert).up_proj.weight"
          guard let gate = sourceArrays[gateKey] else { throw RescoreError.missingTensor(gateKey) }
          guard let up = sourceArrays[upKey] else { throw RescoreError.missingTensor(upKey) }
          batch.append(concatenated([gate, up], axis: -2))
        } else {
          let key = "\(prefix).\(expert).down_proj.weight"
          guard let down = sourceArrays[key] else { throw RescoreError.missingTensor(key) }
          batch.append(down)
        }
      }

      let sourceBatch = MLX.stacked(batch)
      MLX.eval(sourceBatch)
      let result = searched(sourceBatch)
      weightParts.append(result.weight)
      scaleParts.append(result.scales)
      biasParts.append(result.biases)
      print(
        "    layer \(routed.layer) \(routed.projection): experts \(start)...\(end - 1)"
      )
      Memory.clearCache()
    }

    let result = QuantizedArrays(
      weight: concatenated(weightParts, axis: 0),
      scales: concatenated(scaleParts, axis: 0),
      biases: concatenated(biasParts, axis: 0)
    )
    MLX.eval(result.weight, result.scales, result.biases)
    Stream.defaultStream(Device.defaultDevice()).synchronize()
    return result
  }

  private func replacementEntries(
    _ replacement: QuantizedArrays,
    module: String
  ) -> [(String, MLXArray)] {
    [
      ("\(module).weight", replacement.weight),
      ("\(module).scales", replacement.scales),
      ("\(module).biases", replacement.biases),
    ]
  }

  private func install(
    _ value: MLXArray,
    key: String,
    arrays: inout [String: MLXArray]
  ) throws {
    guard let templateValue = arrays[key] else {
      throw RescoreError.incompatibleTemplate("\(key) is not in its declared shard")
    }
    guard value.shape == templateValue.shape,
      value.dtype == templateValue.dtype,
      value.nbytes == templateValue.nbytes
    else {
      throw RescoreError.incompatibleTemplate(
        "\(key) expected shape=\(templateValue.shape) dtype=\(templateValue.dtype) bytes=\(templateValue.nbytes), "
          + "got shape=\(value.shape) dtype=\(value.dtype) bytes=\(value.nbytes)"
      )
    }
    arrays[key] = value
  }

  private func verifyTemplateIdentity(
    sourceArrays: [String: MLXArray],
    templateURL: URL,
    templateIndex: SafetensorsIndex
  ) throws {
    let checks: [(module: String, bits: Int, source: String, expert: Int?)] = [
      (
        "language_model.model.layers.1.mlp.shared_expert.down_proj",
        4,
        "model.layers.1.mlp.shared_expert.down_proj.weight",
        nil
      ),
      (
        "language_model.model.layers.1.mlp.gate.proj",
        8,
        "model.layers.1.mlp.gate.weight",
        nil
      ),
      (
        "language_model.model.layers.1.mlp.switch_mlp.down_proj",
        4,
        "model.layers.1.mlp.experts.0.down_proj.weight",
        0
      ),
    ]

    for check in checks {
      let weightKey = "\(check.module).weight"
      guard templateIndex.weightMap[weightKey] != nil else {
        throw RescoreError.incompatibleTemplate("missing identity check key \(weightKey)")
      }
      guard let source = sourceArrays[check.source] else {
        throw RescoreError.missingTensor(check.source)
      }
      let standard = MLX.quantized(
        source, groupSize: 64, bits: check.bits, mode: .affine, stream: .cpu)
      guard let standardBiases = standard.biases else {
        throw RescoreError.incompatibleTemplate("standard affine quantization omitted biases")
      }
      let expectedWeight = try templateArray(
        templateURL: templateURL,
        index: templateIndex,
        key: weightKey,
        expert: check.expert
      )
      let expectedScales = try templateArray(
        templateURL: templateURL,
        index: templateIndex,
        key: "\(check.module).scales",
        expert: check.expert
      )
      let expectedBiases = try templateArray(
        templateURL: templateURL,
        index: templateIndex,
        key: "\(check.module).biases",
        expert: check.expert
      )
      guard arraysEqual(standard.wq, expectedWeight),
        arraysEqual(standard.scales, expectedScales),
        arraysEqual(standardBiases, expectedBiases)
      else {
        throw RescoreError.incompatibleTemplate(
          "\(check.module) does not match standard \(check.bits)-bit quantization of the source"
        )
      }
      Memory.clearCache()
    }

    let fusedModule = "language_model.model.layers.1.mlp.switch_mlp.gate_up_proj"
    guard templateIndex.weightMap["\(fusedModule).weight"] != nil else {
      throw RescoreError.incompatibleTemplate("missing fused gate/up identity check")
    }
    let gateKey = "model.layers.1.mlp.experts.0.gate_proj.weight"
    let upKey = "model.layers.1.mlp.experts.0.up_proj.weight"
    guard let gate = sourceArrays[gateKey] else { throw RescoreError.missingTensor(gateKey) }
    guard let up = sourceArrays[upKey] else { throw RescoreError.missingTensor(upKey) }
    let standardGate = MLX.quantized(
      gate, groupSize: 64, bits: 4, mode: .affine, stream: .cpu)
    let standardUp = MLX.quantized(
      up, groupSize: 64, bits: 4, mode: .affine, stream: .cpu)
    guard let gateBiases = standardGate.biases, let upBiases = standardUp.biases else {
      throw RescoreError.incompatibleTemplate("standard fused quantization omitted biases")
    }
    let standardFused = QuantizedArrays(
      weight: concatenated([standardGate.wq, standardUp.wq], axis: -2),
      scales: concatenated([standardGate.scales, standardUp.scales], axis: -2),
      biases: concatenated([gateBiases, upBiases], axis: -2)
    )
    guard arraysEqual(
      standardFused.weight,
      try templateArray(
        templateURL: templateURL,
        index: templateIndex,
        key: "\(fusedModule).weight",
        expert: 0
      )
    ), arraysEqual(
      standardFused.scales,
      try templateArray(
        templateURL: templateURL,
        index: templateIndex,
        key: "\(fusedModule).scales",
        expert: 0
      )
    ), arraysEqual(
      standardFused.biases,
      try templateArray(
        templateURL: templateURL,
        index: templateIndex,
        key: "\(fusedModule).biases",
        expert: 0
      )
    ) else {
      throw RescoreError.incompatibleTemplate(
        "fused routed gate/up layout does not match source expert order"
      )
    }
    Memory.clearCache()
  }

  private func templateArray(
    templateURL: URL,
    index: SafetensorsIndex,
    key: String,
    expert: Int?
  ) throws -> MLXArray {
    guard let shard = index.weightMap[key] else {
      throw RescoreError.incompatibleTemplate("missing index entry for \(key)")
    }
    let arrays = try loadArrays(
      url: templateURL.appendingPathComponent(shard), stream: .cpu)
    guard let value = arrays[key] else {
      throw RescoreError.incompatibleTemplate("missing \(key) in declared shard")
    }
    if let expert { return value[expert] }
    return value
  }

  private func arraysEqual(_ lhs: MLXArray, _ rhs: MLXArray) -> Bool {
    let equal = MLX.arrayEqual(lhs, rhs)
    MLX.eval(equal)
    Stream.defaultStream(Device.defaultDevice()).synchronize()
    return equal.item(Bool.self)
  }

  private func copySidecars(from templateURL: URL, to destinationURL: URL) throws {
    for item in try FileManager.default.contentsOfDirectory(
      at: templateURL,
      includingPropertiesForKeys: [.isDirectoryKey],
      options: [.skipsHiddenFiles]
    ) {
      if item.pathExtension == "safetensors"
        || item.lastPathComponent == "model.safetensors.index.json"
        || item.lastPathComponent == "q4r8-scale-search.json"
      {
        continue
      }
      try FileManager.default.copyItem(
        at: item,
        to: destinationURL.appendingPathComponent(item.lastPathComponent)
      )
    }
  }

  private func validateSourceMappings(
    _ modules: [String],
    sourceWeightMap: [String: String],
    numberOfExperts: Int
  ) throws {
    for module in modules {
      if let routed = Self.routedProjection(module) {
        let prefix = "model.layers.\(routed.layer).mlp.experts"
        for expert in 0 ..< numberOfExperts {
          let projections =
            routed.projection == "gate_up_proj"
            ? ["gate_proj", "up_proj"] : ["down_proj"]
          for projection in projections {
            let key = "\(prefix).\(expert).\(projection).weight"
            guard sourceWeightMap[key] != nil else { throw RescoreError.missingTensor(key) }
          }
        }
      } else {
        let key = try Self.directSourceWeightKey(for: module)
        guard sourceWeightMap[key] != nil else { throw RescoreError.missingTensor(key) }
      }
    }
  }

  private func decodeIndex(at url: URL) throws -> SafetensorsIndex {
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw RescoreError.invalidInput("missing safetensors index: \(url.path)")
    }
    return try JSONDecoder().decode(SafetensorsIndex.self, from: Data(contentsOf: url))
  }

  private func validateDirectory(_ url: URL, label: String) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw RescoreError.invalidInput("\(label) directory does not exist: \(url.path)")
    }
  }

  private static func isQ8Router(_ module: String) -> Bool {
    module.hasSuffix(".mlp.gate.proj")
  }

  private static func isStandardEmbedding(_ module: String) -> Bool {
    module == "language_model.model.embed_tokens"
  }

  private static func directSourceWeightKey(for module: String) throws -> String {
    let prefix = "language_model."
    guard module.hasPrefix(prefix), routedProjection(module) == nil else {
      throw RescoreError.incompatibleTemplate("cannot map direct Q4 module \(module)")
    }
    return String(module.dropFirst(prefix.count)) + ".weight"
  }

  private static func routedProjection(
    _ module: String
  ) -> (layer: Int, projection: String)? {
    let parts = module.split(separator: ".")
    guard parts.count == 7,
      parts[0] == "language_model",
      parts[1] == "model",
      parts[2] == "layers",
      let layer = Int(parts[3]),
      parts[4] == "mlp",
      parts[5] == "switch_mlp",
      parts[6] == "down_proj" || parts[6] == "gate_up_proj"
    else { return nil }
    return (layer, String(parts[6]))
  }
}
