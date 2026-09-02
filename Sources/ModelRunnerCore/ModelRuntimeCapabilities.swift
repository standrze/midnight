import Foundation

enum MistralRuntimeFamily: String, Equatable, Sendable {
  case mistral
  case mistral3
  case mixtral
}

/// Capabilities derived from checkpoint metadata rather than a repository or
/// folder name. This keeps family-wide runtime policy independent from any one
/// Mistral checkpoint while avoiding accidental activation for Voxtral audio
/// models or unrelated architectures.
struct ModelRuntimeCapabilities: Equatable, Sendable {
  let mistralFamily: MistralRuntimeFamily?

  static let none = Self(mistralFamily: nil)

  var supportsMistralConversationPrefixCache: Bool {
    mistralFamily != nil
  }

  static func load(from modelDirectory: URL) throws -> Self {
    let configurationURL = modelDirectory.appendingPathComponent("config.json")
    return try decode(Data(contentsOf: configurationURL))
  }

  static func decode(_ data: Data) throws -> Self {
    guard
      let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return .none }

    let rootType = normalized(root["model_type"] as? String)
    if isAudioFamily(rootType) { return .none }
    let architectures = (root["architectures"] as? [String]) ?? []
    if architectures.contains(where: { isAudioFamily(normalized($0)) }) {
      return .none
    }
    if let family = family(for: rootType) {
      return Self(mistralFamily: family)
    }

    if let textConfiguration = root["text_config"] as? [String: Any] {
      let textType = normalized(textConfiguration["model_type"] as? String)
      if let family = family(for: textType) {
        return Self(mistralFamily: family)
      }
    }

    for architecture in architectures {
      let normalizedArchitecture = normalized(architecture)
      if normalizedArchitecture.contains("mixtral") {
        return Self(mistralFamily: .mixtral)
      }
      if normalizedArchitecture.contains("mistral3")
        || normalizedArchitecture.contains("ministral3")
      {
        return Self(mistralFamily: .mistral3)
      }
      if normalizedArchitecture.contains("mistral") {
        return Self(mistralFamily: .mistral)
      }
    }

    return .none
  }

  private static func family(for modelType: String) -> MistralRuntimeFamily? {
    switch modelType {
    case "mistral": .mistral
    case "mistral3", "mistral3_text", "ministral3": .mistral3
    case "mixtral": .mixtral
    default: nil
    }
  }

  private static func isAudioFamily(_ value: String) -> Bool {
    value.contains("voxtral") || value.contains("audio") || value.contains("speech")
  }

  private static func normalized(_ value: String?) -> String {
    value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
  }
}
