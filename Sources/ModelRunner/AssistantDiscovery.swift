import Foundation
import ModelRunnerCore
import ModelRunnerProtocol

/// Metadata-only discovery. Network access belongs to the load operation, never catalog listing.
struct AssistantDiscovery: Sendable {
    enum Kind: Sendable { case gemma, dflash }
    struct Selection: Sendable {
        let kind: Kind
        let directory: URL
        var repository: String? = nil
    }

    static var defaultDirectory: URL {
        if let path = ProcessInfo.processInfo.environment["MODEL_RUNNER_ASSISTANTS_DIR"], !path.isEmpty {
            return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".midnight/drafters")
    }

    let directory: URL

    func resolve(
        target: URL, settingsDirectory: URL, configuration: Data,
        card: ModelCard?, blockSize: Int?, quantizationBits: Int?
    ) throws -> Selection? {
        let object = try JSONSerialization.jsonObject(with: configuration) as? [String: Any]
        let kind: Kind
        switch object?["model_type"] as? String {
        case "gemma4", "gemma4_text": kind = .gemma
        case "muse_glimmer": kind = .dflash
        default: return nil
        }

        func compatible(_ candidate: URL) -> Bool {
            guard let data = try? ModelLoader.validateModelFolder(candidate) else {
                return false
            }
            return
                (try? Self.validate(
                    kind: kind, target: configuration, assistant: data,
                    blockSize: blockSize, quantizationBits: quantizationBits)) != nil
        }

        // A named pairing is authoritative. Do not substitute another variant when it is missing.
        let references = try Self.references(settingsDirectory: settingsDirectory, card: card)
        if let reference = references.first, references.count == 1 {
            let local = settingsDirectory.appendingPathComponent(reference).standardizedFileURL
            let expanded = NSString(string: reference).expandingTildeInPath
            let absolute = URL(fileURLWithPath: expanded)
            let candidates: [URL]
            if NSString(string: expanded).isAbsolutePath {
                candidates = [absolute]
            } else {
                candidates = [
                    local, directory.appendingPathComponent(reference.replacingOccurrences(of: "/", with: "--")),
                    ModelCatalog.defaultDirectory().appendingPathComponent(
                        reference.replacingOccurrences(of: "/", with: "--")),
                ]
            }
            for candidate in candidates where FileManager.default.fileExists(atPath: candidate.path) {
                guard compatible(candidate) else {
                    throw ModelLoadingError(
                        "The model card's assistant is incomplete or incompatible: \(candidate.path)")
                }
                return Selection(kind: kind, directory: candidate)
            }
            if !reference.hasPrefix("/"), !reference.hasPrefix("~"),
                (try? DownloadPlan.repository(reference)) != nil
            {
                return Selection(
                    kind: kind,
                    directory: directory.appendingPathComponent(reference.replacingOccurrences(of: "/", with: "--")),
                    repository: reference)
            }
            throw ModelLoadingError("The model card's local assistant does not exist: \(reference)")
        }

        for name in ["drafter", "assistant"] {
            let embedded = target.appendingPathComponent(name)
            if compatible(embedded) {
                return Selection(kind: kind, directory: embedded)
            }
        }
        let children =
            (try? FileManager.default.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        var seen = Set<String>()
        let candidates = children.filter {
            seen.insert($0.resolvingSymlinksInPath().path).inserted && compatible($0)
        }
        if candidates.count == 1 {
            return Selection(kind: kind, directory: candidates[0])
        }

        // Distinguish the original and QAT Gemma assistants when both share tensor dimensions.
        // Conversion provenance is stronger than the locally assigned folder name.
        let metadata = Self.json(settingsDirectory.appendingPathComponent(ModelCard.filename))
        let identity = (metadata?["source_model"] as? String) ?? target.lastPathComponent
        let matching = candidates.filter {
            Self.gemmaIdentity($0.lastPathComponent) == Self.gemmaIdentity(identity)
                && Self.gemmaIdentity(identity) != nil
        }
        if matching.count == 1 {
            return Selection(kind: kind, directory: matching[0])
        }
        // Ambiguous candidates require a card reference or explicit path; directory order is not policy.
        return nil
    }

    static func validate(
        kind: Kind, target: Data, assistant: Data,
        blockSize: Int?, quantizationBits: Int?
    ) throws {
        switch kind {
        case .gemma:
            _ = try GemmaAssistantCompatibility.validate(
                target: target, assistant: assistant,
                blockSize: blockSize, quantizationBits: quantizationBits)
        case .dflash:
            _ = try MuseGlimmerDFlashCompatibility.validate(target: target, assistant: assistant, blockSize: blockSize)
        }
    }

    static func references(settingsDirectory: URL, card: ModelCard?) throws -> [String] {
        if let reference = card?.assistantModel?.trimmingCharacters(in: .whitespacesAndNewlines), !reference.isEmpty {
            return [reference]
        }
        guard let text = readText(settingsDirectory.appendingPathComponent("README.md")) else {
            return []
        }
        // Read data from the downloaded Hugging Face card, never execute its examples.
        let assignments = matches(
            #"(?im)\b(?:ASSISTANT_MODEL_ID|assistant_model_name|assistant_model|drafter_model|dflash_model)\s*[:=]\s*["']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)["']"#,
            in: text)
        let links = matches(#"https://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"#, in: text)
            .filter { $0.lowercased().contains("assistant") || $0.lowercased().contains("dflash") }
        return Array(Set(assignments.isEmpty ? links : assignments)).sorted()
    }

    private static func gemmaIdentity(_ text: String) -> String? {
        matches(#"(gemma-4-[a-z0-9]+(?:-[a-z0-9]+)?-it(?:-qat-q4_0)?)"#, in: text.lowercased()).first
    }

    private static func matches(_ pattern: String, in text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }
        return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap {
            Range($0.range(at: 1), in: text).map { String(text[$0]) }
        }
    }

    private static func readText(_ url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            return nil
        }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: 1_048_577), data.count <= 1_048_576 else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    private static func json(_ url: URL) -> [String: Any]? {
        guard let text = readText(url) else {
            return nil
        }
        return (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any]
    }
}
