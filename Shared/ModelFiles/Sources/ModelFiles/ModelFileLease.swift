import Foundation

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

/// Advisory leases are acquired only while loading or managing models. A loaded
/// backend owns its shared leases until every request and producer has drained.
/// Persistent lock files must never be unlinked: removing/re-downloading a model
/// must not let two processes lock different inodes for the same folder name.
public final class ModelFileUsage: Sendable {
    public let leases: [ModelFileLease]
    /// Groups leases owned by one loaded model backend.
    public init(_ leases: [ModelFileLease]) {
        self.leases = leases
    }
}

/// Owns a process-scoped advisory file lock until the lease is released.
public final class ModelFileLease: Sendable {
    private let descriptor: Int32
    private let ownerProcessID: pid_t

    fileprivate init(directory: Int32, name: String, shared: Bool) throws {
        let descriptor = openat(directory, name, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
        guard descriptor >= 0 else {
            throw ModelFileLeaseError.leaseUnavailable
        }
        var metadata = stat()
        guard fstat(descriptor, &metadata) == 0,
            metadata.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
            metadata.st_uid == geteuid(), metadata.st_nlink == 1,
            metadata.st_mode & mode_t(0o077) == 0
        else {
            close(descriptor)
            throw ModelFileLeaseError.leaseUnavailable
        }
        var result: Int32
        repeat {
            result = flock(descriptor, (shared ? LOCK_SH : LOCK_EX) | LOCK_NB)
        } while result != 0 && errno == EINTR
        guard result == 0 else {
            let busy = errno == EWOULDBLOCK || errno == EAGAIN
            close(descriptor)
            throw busy ? ModelFileLeaseError.inUse : ModelFileLeaseError.leaseUnavailable
        }
        self.descriptor = descriptor
        self.ownerProcessID = getpid()
    }

    deinit {
        // A fork/dup shares the open file description. O_CLOEXEC closes an
        // inherited descriptor only at exec, so close alone can leave a lease
        // alive briefly after this owner is gone. Release it explicitly here.
        // A forked child's teardown must never unlock its still-live parent.
        if getpid() == ownerProcessID {
            var result: Int32
            repeat {
                result = flock(descriptor, LOCK_UN)
            } while result != 0 && errno == EINTR
        }
        close(descriptor)
    }
}

/// Open relative to checked directory descriptors, refusing symlink lock stores
/// and files. A model root may have aliased ancestors; its canonical directory
/// still yields the same on-disk leases for every cooperating process.
public final class ModelFileLeaseDirectory {
    private let descriptor: Int32

    /// Opens a checked lease directory beneath the supplied model root.
    public init(root: URL) throws {
        let rootDescriptor = open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard rootDescriptor >= 0 else {
            throw ModelFileLeaseError.unsafeDirectory
        }
        defer { close(rootDescriptor) }
        if mkdirat(rootDescriptor, ".leases", mode_t(0o700)) != 0 && errno != EEXIST {
            throw ModelFileLeaseError.leaseUnavailable
        }
        let descriptor = openat(rootDescriptor, ".leases", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw ModelFileLeaseError.leaseUnavailable
        }
        var metadata = stat()
        guard fstat(descriptor, &metadata) == 0, metadata.st_uid == geteuid(),
            metadata.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR),
            metadata.st_mode & mode_t(0o077) == 0
        else {
            close(descriptor)
            throw ModelFileLeaseError.leaseUnavailable
        }
        self.descriptor = descriptor
    }

    deinit {
        close(descriptor)
    }

    /// Takes a shared or exclusive lock on the model-root lease file.
    public func root(shared: Bool) throws -> ModelFileLease {
        try ModelFileLease(directory: descriptor, name: ".root.lock", shared: shared)
    }

    /// Takes a shared or exclusive lock for one validated model name.
    public func model(_ name: String, shared: Bool) throws -> ModelFileLease {
        try Self.validateName(name)
        return try ModelFileLease(directory: descriptor, name: name, shared: shared)
    }
}

/// A model lease is busy, unsafe to open, or invalid for this lock store.
public enum ModelFileLeaseError: Error, Sendable {
    case leaseUnavailable, inUse, unsafeDirectory, invalidName, notFound
}

extension ModelFileLeaseDirectory {
    /// Rejects names that cannot safely identify a model lease file.
    public static func validateName(_ name: String) throws {
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        guard !name.isEmpty, !name.hasPrefix("."), !name.hasPrefix("-"),
            name.unicodeScalars.allSatisfy(allowed.contains)
        else {
            throw ModelFileLeaseError.invalidName
        }
    }
}

extension ModelFileUsage {
    /// A separate process must acquire its own leases before opening weights.
    /// Never transfer/duplicate a parent's open file description: parent teardown
    /// explicitly unlocks that description, including any inherited descriptors.
    public static func acquire(protectedDirectories: [URL], root: URL) throws -> ModelFileUsage {
        let canonicalRoot = root.resolvingSymlinksInPath().pathComponents
        let selected = protectedDirectories.map { $0.standardizedFileURL.resolvingSymlinksInPath() }
        let relevant = selected.filter {
            $0.pathComponents.starts(with: canonicalRoot) || canonicalRoot.starts(with: $0.pathComponents)
        }
        guard !relevant.isEmpty else {
            return ModelFileUsage([])
        }
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: root.path),
            attributes[.type] as? FileAttributeType == .typeDirectory
        else {
            throw ModelFileLeaseError.unsafeDirectory
        }
        let locks = try ModelFileLeaseDirectory(root: root)
        var leases: [ModelFileLease] = []
        if relevant.contains(where: { canonicalRoot.starts(with: $0.pathComponents) }) {
            leases.append(try locks.root(shared: true))
        }
        let names = Set(
            relevant.compactMap { path -> String? in
                let components = path.pathComponents
                guard components.count > canonicalRoot.count else {
                    return nil
                }
                let name = components[canonicalRoot.count]
                guard (try? ModelFileLeaseDirectory.validateName(name)) != nil else {
                    return nil
                }
                return name
            })
        for name in names.sorted() {
            leases.append(try locks.model(name, shared: true))
        }
        for path in relevant where !FileManager.default.fileExists(atPath: path.path) {
            throw ModelFileLeaseError.notFound
        }
        return ModelFileUsage(leases)
    }
}
