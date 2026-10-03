import Foundation
import ModelFiles

// Keep Midnight's management error contract while sharing the exact lease
// implementation with optional workers. Every process acquires independent FDs.
typealias ManagedModelFileLease = ModelFileLease
typealias ManagedModelUsage = ModelFileUsage

final class ManagedModelLeaseDirectory {
    private let shared: ModelFileLeaseDirectory
    init(root: URL) throws { shared = try managedLeaseErrors { try ModelFileLeaseDirectory(root: root) } }
    func root(shared: Bool) throws -> ManagedModelFileLease {
        try managedLeaseErrors { try self.shared.root(shared: shared) }
    }
    func model(_ name: String, shared: Bool) throws -> ManagedModelFileLease {
        try managedLeaseErrors { try self.shared.model(name, shared: shared) }
    }
}

func managedLeaseErrors<T>(_ operation: () throws -> T) throws -> T {
    do { return try operation() } catch let error as ModelFileLeaseError {
        switch error {
        case .leaseUnavailable: throw ModelRemovalError.leaseUnavailable
        case .inUse: throw ModelRemovalError.inUse
        case .unsafeDirectory: throw ModelRemovalError.unsafeDirectory
        case .invalidName: throw ModelRemovalError.invalidName
        case .notFound: throw ModelRemovalError.notFound
        }
    }
}
