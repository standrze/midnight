import Foundation
import ModelFiles

struct ManagedModelDownload: Codable, Sendable {
    let id: String
    let repository: String
    let revision: String
    let inUse: Bool
}

struct ManagedModelDownloadList: Codable, Sendable {
    let object = "list"
    let data: [ManagedModelDownload]
}

struct ManagedModelRemoval: Codable, Sendable {
    let id: String
    let object = "model"
    let deleted: Bool
}

enum ModelRemovalError: LocalizedError {
    case invalidName, notFound, unmanaged, inUse, unsafeDirectory, leaseUnavailable
    case cleanupFailed(String)

    var errorDescription: String? {
        switch self {
        case .invalidName:
            "Use a downloaded model's folder name, without paths. List names with midnight remove --list."
        case .notFound: "That downloaded model does not exist."
        case .unmanaged:
            "Only checkpoints created by midnight download can be removed. This folder has no valid download record."
        case .inUse:
            "This model or one of its files is in use by a Midnight instance. Unload it there before removing it."
        case .unsafeDirectory:
            "Removal requires a real directory directly inside Midnight's model download folder; symbolic links are not accepted."
        case .leaseUnavailable:
            "Midnight could not safely acquire the model file lease. Check access to the model download folder."
        case .cleanupFailed(let path):
            "The model was detached, but file cleanup failed. Its remaining files are at \(path)."
        }
    }

    var code: String {
        switch self {
        case .invalidName: "invalid_model_name"
        case .notFound: "model_not_found"
        case .unmanaged: "model_not_managed"
        case .inUse: "model_in_use"
        case .unsafeDirectory: "unsafe_model_directory"
        case .leaseUnavailable: "model_lease_unavailable"
        case .cleanupFailed: "model_cleanup_failed"
        }
    }
}

/// Explicit filesystem management only. Never called while preparing or decoding tokens.
struct ManagedModelDownloads: Sendable {
    let root: URL

    init(
        root: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".midnight/models", isDirectory: true)
    ) {
        self.root = root.standardizedFileURL
    }

    func list(protectedDirectories: [URL]) throws -> ManagedModelDownloadList {
        guard FileManager.default.fileExists(atPath: root.path) else {
            return ManagedModelDownloadList(data: [])
        }
        try requireDirectory(root)
        let children = try FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        let models = children.compactMap { child -> ManagedModelDownload? in
            guard let record = try? record(named: child.lastPathComponent) else {
                return nil
            }
            return ManagedModelDownload(
                id: child.lastPathComponent, repository: record.repository,
                revision: record.revision,
                inUse: overlaps(child, protectedDirectories) || isLeased(named: child.lastPathComponent))
        }.sorted { $0.id < $1.id }
        return ManagedModelDownloadList(data: models)
    }

    /// Resolve only explicitly selected paths: ordinary loads never scan the
    /// model catalog. Ancestors of the managed root use one aggregate lease.
    func acquireUsage(protectedDirectories: [URL]) throws -> ManagedModelUsage {
        try managedLeaseErrors {
            try ModelFileUsage.acquire(protectedDirectories: protectedDirectories, root: root)
        }
    }

    func detach(named name: String, protectedDirectories: [URL]) throws -> DetachedModelDownload {
        try Self.validateName(name)
        guard FileManager.default.fileExists(atPath: root.path) else {
            throw ModelRemovalError.notFound
        }
        try requireDirectory(root)
        let locks = try ManagedModelLeaseDirectory(root: root)
        // Lock before reading provenance or renaming. The aggregate lock covers
        // a backend using the managed root/ancestor; per-model locks let normal
        // loads and unrelated downloaded models remain independent.
        let rootLease = try locks.root(shared: false)
        defer { withExtendedLifetime(rootLease) {} }
        let modelLease = try locks.model(name, shared: false)
        _ = try record(named: name)
        let directory = root.appendingPathComponent(name, isDirectory: true)
        guard !overlaps(directory, protectedDirectories) else {
            throw ModelRemovalError.inUse
        }
        let detached = root.appendingPathComponent(".remove-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.moveItem(at: directory, to: detached)
        return DetachedModelDownload(name: name, directory: detached, lease: modelLease)
    }

    private func isLeased(named name: String) -> Bool {
        // Probe without waiting, and fail closed if lease storage is unavailable.
        do {
            let locks = try ManagedModelLeaseDirectory(root: root)
            let rootLease = try locks.root(shared: false)
            let modelLease = try locks.model(name, shared: false)
            withExtendedLifetime((rootLease, modelLease)) {}
            return false
        } catch {
            return true
        }
    }

    static func validateName(_ name: String) throws {
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        guard !name.isEmpty, !name.hasPrefix("."), !name.hasPrefix("-"),
            name.unicodeScalars.allSatisfy(allowed.contains)
        else {
            throw ModelRemovalError.invalidName
        }
    }

    private struct Record: Decodable {
        let repository: String
        let revision: String
    }

    private func record(named name: String) throws -> Record {
        try Self.validateName(name)
        guard FileManager.default.fileExists(atPath: root.path) else {
            throw ModelRemovalError.notFound
        }
        try requireDirectory(root)
        let directory = root.appendingPathComponent(name, isDirectory: true)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw ModelRemovalError.notFound
        }
        try requireDirectory(directory)
        // Foundation URL equality can distinguish a trailing directory slash
        // on Linux even when both URLs resolve to the same directory. Compare
        // canonical filesystem components, as the active-path check does.
        guard
            directory.resolvingSymlinksInPath().deletingLastPathComponent().pathComponents
                == root.resolvingSymlinksInPath().pathComponents
        else {
            throw ModelRemovalError.unsafeDirectory
        }
        let marker = directory.appendingPathComponent("midnight-download.json")
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: marker.path),
            attributes[.type] as? FileAttributeType == .typeRegular,
            (attributes[.size] as? NSNumber)?.intValue ?? Int.max <= 16 * 1024,
            let data = try? Data(contentsOf: marker),
            let record = try? JSONDecoder().decode(Record.self, from: data),
            (try? DownloadPlan.repository(record.repository)) != nil,
            record.revision.count == 40, record.revision.allSatisfy(\.isHexDigit)
        else {
            throw ModelRemovalError.unmanaged
        }
        return record
    }

    private func requireDirectory(_ url: URL) throws {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
            attributes[.type] as? FileAttributeType == .typeDirectory
        else {
            throw ModelRemovalError.unsafeDirectory
        }
    }

    private func overlaps(_ directory: URL, _ protectedDirectories: [URL]) -> Bool {
        let candidate = directory.standardizedFileURL.resolvingSymlinksInPath().pathComponents
        return protectedDirectories.contains {
            let active = $0.standardizedFileURL.resolvingSymlinksInPath().pathComponents
            return candidate.starts(with: active) || active.starts(with: candidate)
        }
    }
}

struct DetachedModelDownload: Sendable {
    let name: String
    let directory: URL
    let lease: ManagedModelFileLease

    func finish() throws -> ManagedModelRemoval {
        defer { withExtendedLifetime(lease) {} }
        do { try FileManager.default.removeItem(at: directory) } catch {
            throw ModelRemovalError.cleanupFailed(directory.path)
        }
        return ManagedModelRemoval(id: name, deleted: true)
    }
}
