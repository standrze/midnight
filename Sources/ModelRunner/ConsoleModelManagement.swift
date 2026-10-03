import Foundation

/// Confirmation state retains the exact deletion target even if discovery changes.
struct ConsoleModelManagement {
    var showingDownloads = false
    var downloadIndex = 0
    private(set) var deletionTarget: String?

    mutating func requestDeletion(_ id: String) { deletionTarget = id }
    mutating func cancelDeletion() { deletionTarget = nil }
    mutating func confirmDeletion() -> String? {
        defer { deletionTarget = nil }
        return deletionTarget
    }

    var prompt: String? {
        deletionTarget.map { "Delete \($0) from disk? y confirm · n/Esc cancel" }
    }
}
