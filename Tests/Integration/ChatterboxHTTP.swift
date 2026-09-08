// Run against an already-running server:
// swift Tests/Integration/ChatterboxHTTP.swift http://127.0.0.1:18081 chatterbox-turbo default /tmp/chatterbox-http
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

struct CheckFailure: Error { let message: String }
func check(_ condition: Bool, _ message: String) throws {
    if !condition { throw CheckFailure(message: message) }
}
let args = CommandLine.arguments
guard args.count == 5 else { fatalError("Expected server URL, model name, voice name, output directory") }
let base = args[1], model = args[2], voice = args[3]
let output = URL(fileURLWithPath: args[4], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let session = URLSession(configuration: .ephemeral)

func speech(_ extra: [String: Any] = [:]) async throws -> (Data, HTTPURLResponse) {
    var body: [String: Any] = ["model": model, "voice": voice, "input": "Hello."]
    body.merge(extra) { _, new in new }
    var request = URLRequest(url: URL(string: base + "/v1/audio/speech")!)
    request.httpMethod = "POST"
    request.timeoutInterval = 180
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    let (data, response) = try await session.data(for: request)
    return (data, response as! HTTPURLResponse)
}

let (models, modelResponse) = try await session.data(from: URL(string: base + "/v1/models")!)
try check((modelResponse as! HTTPURLResponse).statusCode == 200, "model discovery status")
let descriptors = try JSONSerialization.jsonObject(with: models) as! [String: Any]
try check((descriptors["data"] as? [[String: Any]])?.contains(where: { $0["id"] as? String == model }) == true, "model discovery ID")

for format in ["mp3", "wav", "pcm", "flac", "opus", "aac"] {
    // Omitting response_format is the standard OpenAI MP3 default.
    let (data, response) = try await speech(format == "mp3" ? [:] : ["response_format": format])
    try check(response.statusCode == 200, "\(format) status: \(response.statusCode) \(String(decoding: data.prefix(500), as: UTF8.self))")
    try check(!data.isEmpty, "\(format) empty audio")
    let contentType = response.value(forHTTPHeaderField: "Content-Type") ?? ""
    try check(contentType.hasPrefix("audio/") || (format == "pcm" && contentType == "application/octet-stream"), "\(format) content type")
    try data.write(to: output.appendingPathComponent("speech.\(format)"))
    print("PASS \(format)")
}

let (sse, sseResponse) = try await speech(["response_format": "pcm", "stream_format": "sse"])
try check(sseResponse.statusCode == 200, "SSE status")
try check(sseResponse.value(forHTTPHeaderField: "Content-Type")?.hasPrefix("text/event-stream") == true, "SSE content type")
var deltaBytes = 0, done = false
for line in String(decoding: sse, as: UTF8.self).components(separatedBy: "\n") where line.hasPrefix("data: ") {
    let payload = Data(line.dropFirst(6).utf8)
    guard let object = try? JSONSerialization.jsonObject(with: payload) as? [String: Any] else { continue }
    if object["type"] as? String == "speech.audio.delta" {
        guard let encoded = object["audio"] as? String, let audio = Data(base64Encoded: encoded) else { throw CheckFailure(message: "SSE audio") }
        deltaBytes += audio.count
    }
    if object["type"] as? String == "speech.audio.done" { done = true }
}
try check(deltaBytes > 0 && done, "SSE completion")
print("PASS SSE")

for speed in [0.25, 4.0] {
    let (data, response) = try await speech(["response_format": "pcm", "speed": speed])
    try check(response.statusCode == 200 && !data.isEmpty && data.count % 2 == 0, "speed \(speed)")
}
print("PASS speed range")

for (extra, status) in [
    (["model": "missing"] as [String: Any], 404),
    (["voice": "missing"], 400),
    (["instructions": "Whisper"], 400),
    (["speed": 0.1], 400),
    (["input": ""], 400)
] {
    let (data, response) = try await speech(extra)
    try check(response.statusCode == status, "error status \(extra)")
    let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    try check(object?["error"] is [String: Any], "OpenAI error envelope")
}
print("PASS OpenAI errors; all HTTP checks passed")
