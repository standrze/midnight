import Dispatch
import Foundation
import loom

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

/// Redirects process logs while loom writes through a saved terminal descriptor.
/// A dedicated reader drains the pipe even while the main actor is busy loading a model.
final class ConsoleOutput: @unchecked Sendable {
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "midnight.console.logs")
    private let originalOutput: Int32
    private let originalError: Int32
    private let readDescriptor: Int32
    private var source: DispatchSourceRead?
    private var recentBytes = Data()
    private var closed = false

    var backend: ConsoleBackend { ConsoleBackend(descriptor: originalOutput) }

    // weft measures stdout, which is redirected during capture. Query its saved
    // descriptor to retain real terminal dimensions, including SIGWINCH updates.
    var terminalSize: Rect {
        var dimensions = winsize()
        guard ioctl(originalOutput, UInt(TIOCGWINSZ), &dimensions) == 0,
            dimensions.ws_col > 0, dimensions.ws_row > 0
        else {
            return Rect(width: 80, height: 24)
        }
        return Rect(width: Int(dimensions.ws_col), height: Int(dimensions.ws_row))
    }

    init() throws {
        fflush(nil)
        let output = dup(STDOUT_FILENO)
        let error = dup(STDERR_FILENO)
        var descriptors: [Int32] = [-1, -1]
        guard output >= 0, error >= 0, pipe(&descriptors) == 0 else {
            if output >= 0 {
                close(output)
            }
            if error >= 0 {
                close(error)
            }
            throw POSIXError(.EIO)
        }
        originalOutput = output
        originalError = error
        readDescriptor = descriptors[0]
        _ = fcntl(output, F_SETFD, FD_CLOEXEC)
        _ = fcntl(error, F_SETFD, FD_CLOEXEC)
        _ = fcntl(readDescriptor, F_SETFD, FD_CLOEXEC)
        guard fcntl(readDescriptor, F_SETFL, O_NONBLOCK) != -1,
            dup2(descriptors[1], STDOUT_FILENO) != -1,
            dup2(descriptors[1], STDERR_FILENO) != -1
        else {
            dup2(output, STDOUT_FILENO)
            dup2(error, STDERR_FILENO)
            close(output)
            close(error)
            close(descriptors[0])
            close(descriptors[1])
            throw POSIXError(.EIO)
        }
        close(descriptors[1])
        let reader = DispatchSource.makeReadSource(fileDescriptor: readDescriptor, queue: queue)
        reader.setEventHandler { [weak self] in self?.drain() }
        let readPipe = readDescriptor
        reader.setCancelHandler { close(readPipe) }
        source = reader
        reader.resume()
    }

    deinit { restore() }

    func lines() -> [String] {
        fflush(nil)
        lock.lock()
        defer { lock.unlock() }
        return String(decoding: recentBytes, as: UTF8.self)
            .split(whereSeparator: \.isNewline).suffix(6).map(String.init)
    }

    /// Restore output before closing weft's session so its exit sequences reach the terminal.
    func restore() {
        guard !closed else {
            return
        }
        closed = true
        fflush(nil)
        dup2(originalOutput, STDOUT_FILENO)
        dup2(originalError, STDERR_FILENO)
        // Synchronize with the reader before releasing its descriptor and our saved output.
        queue.sync { drain() }
        source?.cancel()
        source = nil
        close(originalOutput)
        close(originalError)
    }

    private func drain() {
        var bytes = [UInt8](repeating: 0, count: 8192)
        while true {
            let count = read(readDescriptor, &bytes, bytes.count)
            if count < 0, errno == EINTR {
                continue
            }
            guard count > 0 else {
                return
            }
            lock.lock()
            recentBytes.append(contentsOf: bytes.prefix(count))
            if recentBytes.count > 32_768 {
                recentBytes.removeFirst(recentBytes.count - 32_768)
            }
            lock.unlock()
        }
    }
}

/// Writes complete ANSI frames without mixing them with captured stdout/stderr logs.
struct ConsoleBackend: Backend {
    let descriptor: Int32

    mutating func write(_ data: Data) throws {
        try data.withUnsafeBytes { bytes in
            var offset = 0
            while offset < bytes.count {
                #if canImport(Darwin)
                    let count = Darwin.write(descriptor, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
                #else
                    let count = Glibc.write(descriptor, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
                #endif
                if count < 0, errno == EINTR {
                    continue
                }
                guard count > 0 else {
                    throw POSIXError(.EIO)
                }
                offset += count
            }
        }
    }
}
