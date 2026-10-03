import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

struct RemoveModelCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "remove",
        abstract: "Delete a managed download from disk through a running local Midnight listener.")
    @Argument(help: "Downloaded folder name (not a path or served model alias)") var model: String?
    @OptionGroup var listing: ModelListOptions
    @OptionGroup var connection: ModelControlOptions

    mutating func validate() throws {
        guard listing.list != (model != nil) else {
            throw ValidationError("Use midnight remove NAME or midnight remove --list.")
        }
    }

    mutating func run() async throws {
        if listing.list {
            let data = try await LocalModelControl.request(endpoint: connection.endpoint, operation: "downloads")
            let downloads = try JSONDecoder().decode(ManagedModelDownloadList.self, from: data)
            if downloads.data.isEmpty {
                print("No managed downloads.")
            }
            for model in downloads.data {
                print("\(model.id)  \(model.repository)\(model.inUse ? "  (in use; unload first)" : "")")
            }
        } else if let model {
            let data = try await LocalModelControl.request(
                endpoint: connection.endpoint, operation: "remove",
                body: JSONEncoder().encode(["model": model]))
            let result = try JSONDecoder().decode(ManagedModelRemoval.self, from: data)
            guard result.deleted, result.id == model else {
                throw ValidationError("Invalid removal response.")
            }
            print("Removed \(result.id) from disk.")
        }
    }
}

struct UnloadModelCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "unload",
        abstract: "Unload the active model from memory, keeping its downloaded files.")
    @OptionGroup var connection: ModelControlOptions

    mutating func run() async throws {
        _ = try await LocalModelControl.request(
            endpoint: connection.endpoint, operation: "unload", body: Data("{}".utf8))
        print("Unload accepted. Files are kept; removal is available after outstanding requests finish.")
    }
}

enum LocalModelControl {
    static func request(endpoint: String, operation: String, body: Data? = nil) async throws -> Data {
        guard var parts = URLComponents(string: endpoint), parts.scheme == "http",
            ["localhost", "127.0.0.1", "[::1]", "::1"].contains(parts.host ?? ""),
            parts.user == nil, parts.password == nil, parts.query == nil, parts.fragment == nil,
            ["", "/", "/v1", "/v1/"].contains(parts.path)
        else {
            throw ValidationError("Model management requires a local Midnight base URL, such as http://127.0.0.1:8080.")
        }
        parts.path = "/v1/runtime/\(operation)"
        guard let url = parts.url else {
            throw ValidationError("Invalid endpoint URL.")
        }
        var request = URLRequest(url: url)
        request.httpMethod = body == nil ? "GET" : "POST"
        request.httpBody = body
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let session = URLSession(configuration: .ephemeral, delegate: ControlNoRedirects(), delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else {
            throw ValidationError("Invalid control response.")
        }
        guard (200..<300).contains(response.statusCode) else {
            struct Envelope: Decodable {
                struct Detail: Decodable { let message: String }
                let error: Detail
            }
            let message =
                (try? JSONDecoder().decode(Envelope.self, from: data).error.message)
                ?? "Midnight returned HTTP \(response.statusCode). A running local listener with model management support is required."
            throw ValidationError(message)
        }
        return data
    }
}

private final class ControlNoRedirects: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}
