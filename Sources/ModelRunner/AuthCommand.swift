import ArgumentParser
import Foundation
import HuggingFace

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif
#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

struct AuthCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "auth", abstract: "Manage the local Hugging Face token.",
        subcommands: [Login.self, Logout.self, Status.self])

    static var tokenURL: URL {
        let env = ProcessInfo.processInfo.environment
        if let path = env["HF_TOKEN_PATH"], !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        if let path = env["HF_HOME"], !path.isEmpty {
            return URL(fileURLWithPath: path).appendingPathComponent("token")
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/token")
    }

    struct Login: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Save a read token locally using Hugging Face's standard token file (0600).")
        mutating func run() async throws {
            guard isatty(STDIN_FILENO) != 0, let input = getpass("Hugging Face read token (hidden): ") else {
                throw ValidationError("Run midnight auth login in an interactive terminal, or set HF_TOKEN.")
            }
            let token = String(cString: input).trimmingCharacters(in: .whitespacesAndNewlines)
            memset(input, 0, strlen(input))
            guard token.hasPrefix("hf_"), !token.contains(where: { $0.isWhitespace }) else {
                throw ValidationError("Expected a Hugging Face token beginning with hf_.")
            }
            var request = URLRequest(url: URL(string: "https://huggingface.co/api/whoami-v2")!)
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            let (_, response) = try await URLSession.shared.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                throw ValidationError("Hugging Face did not accept the token; nothing was saved.")
            }
            let url = AuthCommand.tokenURL
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
            let temp = url.deletingLastPathComponent().appendingPathComponent(".midnight-token-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: temp) }
            guard
                FileManager.default.createFile(
                    atPath: temp.path, contents: Data(token.utf8), attributes: [.posixPermissions: 0o600])
            else {
                throw ValidationError("Could not write token file.")
            }
            guard rename(temp.path, url.path) == 0 else {
                throw ValidationError("Could not install token file.")
            }
            print("Token verified and saved locally. HF_TOKEN takes precedence if set.")
        }
    }

    struct Logout: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Remove the shared Hugging Face token file; environment variables remain unchanged.")
        mutating func run() throws {
            let url = AuthCommand.tokenURL
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            print(
                "Local token file removed. Unset HF_TOKEN and HUGGING_FACE_HUB_TOKEN separately if used; other Hugging Face tools share this file."
            )
        }
    }

    struct Status: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Report whether a token is available without displaying it.")
        mutating func run() async throws {
            let token = await HubClient().bearerToken
            print(
                token?.isEmpty == false
                    ? "Hugging Face token available (not validated)."
                    : "No Hugging Face token found. Run midnight auth login.")
        }
    }
}
