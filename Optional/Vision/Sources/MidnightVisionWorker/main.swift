import Foundation
import ModelFiles
import VisionHTTP
import VisionProtocol

#if os(macOS)
    import Darwin
#endif

@main struct MidnightVisionWorker {
    static func main() async {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            if args == ["--help"] || args.isEmpty {
                print(
                    "Usage: midnight-vision-worker --model /absolute/vision-model --served-model-name NAME --memory-limit-gib N [--port 8001]\nExplicitly loads one vision model in a separate process. Midnight can launch it when that vision model is explicitly selected."
                )
                return
            }
            #if os(macOS)
                var options = [String: String]()
                var index = 0
                let known: Set<String> = [
                    "--model", "--served-model-name", "--memory-limit-gib", "--port", "--managed-root", "--ready-file",
                    "--control-token", "--parent-pid",
                ]
                while index < args.count {
                    let key = args[index]
                    guard known.contains(key), options[key] == nil, index + 1 < args.count else {
                        throw VisionError("invalid_arguments", "Unknown, repeated, or incomplete option: \(key)")
                    }
                    options[key] = args[index + 1]
                    index += 2
                }
                guard let path = options["--model"], path.hasPrefix("/"),
                    let name = options["--served-model-name"], !name.isEmpty, name.utf8.count <= 256,
                    let memory = options["--memory-limit-gib"].flatMap(Double.init), memory.isFinite, memory >= 1,
                    memory <= 256,
                    let port = Int(options["--port"] ?? "8001"), port == 0 || (1024...65535).contains(port)
                else {
                    throw VisionError(
                        "invalid_arguments",
                        "Provide absolute --model, --served-model-name, explicit --memory-limit-gib (1...256), and optional --port (1024...65535)."
                    )
                }
                let readyFile = options["--ready-file"].map { URL(fileURLWithPath: $0) }
                let parentPID = options["--parent-pid"].flatMap(Int32.init)
                let controlToken = options["--control-token"]
                if readyFile != nil || parentPID != nil || controlToken != nil {
                    guard let readyFile, readyFile.path.hasPrefix("/"), let parentPID, parentPID > 1,
                        getppid() == parentPID, let controlToken, controlToken.count >= 32
                    else {
                        throw VisionError(
                            "invalid_arguments",
                            "Managed launch requires an absolute ready file, current parent PID and control token together."
                        )
                    }
                    Task.detached {
                        while getppid() == parentPID {
                            try? await Task.sleep(for: .seconds(1))
                        }
                        // Parent exit leaves nobody to consume this worker. Process
                        // teardown releases GPU resources and this process's leases.
                        exit(0)
                    }
                }
                let directory = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
                var isDirectory: ObjCBool = false
                guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
                    isDirectory.boolValue,
                    FileManager.default.fileExists(atPath: directory.appendingPathComponent("config.json").path),
                    FileManager.default.fileExists(
                        atPath: directory.appendingPathComponent("preprocessor_config.json").path)
                else {
                    throw VisionError(
                        "vision_model_required",
                        "Select an existing local vision checkpoint containing config.json and preprocessor_config.json. Models are never downloaded implicitly."
                    )
                }
                let managedRoot =
                    options["--managed-root"].map { URL(fileURLWithPath: $0, isDirectory: true) }
                    ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
                        ".midnight/models", isDirectory: true)
                // Independent descriptors protect standalone/orphan lifetimes even
                // when the parent's own shared lease is released or its process dies.
                let usage = try ModelFileUsage.acquire(protectedDirectories: [directory], root: managedRoot)
                defer { withExtendedLifetime(usage) {} }
                let backend = try await NativeVisionModel(
                    directory: directory, memoryBytes: Int(memory * 1024 * 1024 * 1024))
                try await VisionHTTPServer(backend: backend, model: name, controlToken: controlToken).run(
                    port: port, readyFile: readyFile)
            #else
                throw VisionError(
                    "unsupported_backend",
                    "Native Midnight vision currently requires macOS with Apple silicon. CUDA vision is not implemented; the text runner remains available."
                )
            #endif
        } catch {
            FileHandle.standardError.write(Data("midnight-vision-worker: \(String(describing: error))\n".utf8))
            exit(1)
        }
    }
}
