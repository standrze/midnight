#if os(macOS)
    import CryptoKit
    import Foundation
    import MLX
    import MLXLMCommon
    import Testing

    @Suite(
        "Opt-in recovered A4B router quantization proof", .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_GEMMA4_ROUTER_PROOF"] == "1"))
    struct Gemma4RouterRecoveryProofTests {
        private static let revision = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"

        @Test("All original BF16 routers reproduce installed searched Q4 bytes before optional Q8 output")
        func reproduceAllRouters() throws {
            let environment = ProcessInfo.processInfo.environment
            let directory = URL(fileURLWithPath: try #require(environment["MIDNIGHT_GEMMA4_ROUTER_RECOVERY_DIR"]))
            let output = URL(fileURLWithPath: try #require(environment["MIDNIGHT_GEMMA4_ROUTER_PROOF_REPORT"]))
            try #require(!FileManager.default.fileExists(atPath: output.path))
            let manifestURL = directory.appendingPathComponent("recovery.json")
            let manifestData = try Data(contentsOf: manifestURL)
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let manifest = try decoder.decode(Recovery.self, from: manifestData)
            try #require(manifest.status == "recovered_pending_quantization_proof")
            try #require(manifest.revision == Self.revision && manifest.routerCount == 30)
            try #require(manifest.sourceCompanionsExact && manifest.installedSelectedTensorsUnchanged)
            try #require(!manifest.candidateCreated)
            let snapshot = try decoder.decode(
                InstalledSnapshot.self,
                from: Data(contentsOf: directory.appendingPathComponent("installed-before.json")))
            let sourceURL = try verifiedFile(
                directory, manifest.sourceArrays, expectedName: "source-routers.safetensors")
            let baselineURL = try verifiedFile(
                directory, manifest.baselineArrays, expectedName: "installed-routers.safetensors")
            try #require(manifest.sourceArrays.tensorCount == 90 && manifest.baselineArrays.tensorCount == 150)
            try #require(
                manifest.sourceArrays.payloadBytes == 21_803_520 && manifest.baselineArrays.payloadBytes == 6_259_200)
            try verifyInstalled(snapshot, directory: URL(fileURLWithPath: manifest.installed))

            var report: [String: Any] = [
                "format": 1, "status": "running", "source_revision": Self.revision,
                "scope": "Exact router conversion proof; no full-model quality or speed conclusion",
                "algorithm": "q4r8_affine_scale_search_ls2", "device": "gpu",
                "bits": 4, "group_size": 64, "metadata_dtype": "BF16",
                "recovery_manifest_sha256": hash(manifestData),
                "source_arrays_sha256": manifest.sourceArrays.sha256,
                "baseline_arrays_sha256": manifest.baselineArrays.sha256,
                "candidate_created": false, "q8_tensor_payload_created": false,
            ]
            try writeReport(report, to: output, initial: true)
            do {
                try Device.withDefaultDevice(.gpu) {
                    let source = try loadArrays(url: sourceURL)
                    let baseline = try loadArrays(url: baselineURL)
                    try verifyKeys(source, baseline: baseline)
                    var rows: [[String: Any]] = []
                    var matchedRouters = 0
                    for layer in 0..<30 {
                        let prefix = "language_model.model.layers.\(layer).router"
                        let original = try array(source, prefix + ".proj.weight", dtype: .bfloat16, shape: [128, 2816])
                        try #require(all(isFinite(original)).item(Bool.self))
                        for (suffix, shape) in [("scale", [2816]), ("per_expert_scale", [128])] {
                            let sourceCompanion = try array(
                                source, prefix + "." + suffix, dtype: .bfloat16, shape: shape)
                            let baselineCompanion = try array(
                                baseline, prefix + "." + suffix, dtype: .bfloat16, shape: shape)
                            try #require(bytes(sourceCompanion) == bytes(baselineCompanion))
                        }

                        // Call the trusted converter on original BF16. Its return already rounds
                        // scales and biases to the source dtype; no extra casts or refits are allowed.
                        let searched = q4AffineScaleSearchQuantized(original, groupSize: 64)
                        eval(searched.weight, searched.scales, searched.biases)
                        let generated = [
                            "weight": searched.weight, "scales": searched.scales, "biases": searched.biases,
                        ]
                        var arrays: [[String: Any]] = []
                        var matches = true
                        for suffix in ["weight", "scales", "biases"] {
                            let value = try #require(generated[suffix])
                            let dtype: DType = suffix == "weight" ? .uint32 : .bfloat16
                            let shape = suffix == "weight" ? [128, 352] : [128, 44]
                            let expected = try array(baseline, prefix + ".proj." + suffix, dtype: dtype, shape: shape)
                            try #require(value.dtype == dtype && value.shape == shape)
                            let actualBytes = bytes(value)
                            let expectedBytes = bytes(expected)
                            let exact = actualBytes == expectedBytes
                            matches = matches && exact
                            arrays.append([
                                "suffix": suffix, "dtype": String(describing: dtype), "shape": shape,
                                "bytes": actualBytes.count, "exact": exact,
                                "actual_sha256": hash(actualBytes), "installed_sha256": hash(expectedBytes),
                                "differing_bytes": zip(actualBytes, expectedBytes).filter { $0.0 != $0.1 }.count,
                            ])
                        }
                        matchedRouters += matches ? 1 : 0
                        rows.append(["layer": layer, "exact": matches, "arrays": arrays])
                        report["routers"] = rows
                        report["matched_routers"] = matchedRouters
                        try writeReport(report, to: output)
                        print("A4B router proof \(layer + 1)/30 exact=\(matches)")
                    }
                    try #require(matchedRouters == 30, "Recovered source failed the installed Q4 reproduction gate")
                    try verifyInstalled(snapshot, directory: URL(fileURLWithPath: manifest.installed))
                    try #require(hash(try Data(contentsOf: sourceURL)) == manifest.sourceArrays.sha256)
                    try #require(hash(try Data(contentsOf: baselineURL)) == manifest.baselineArrays.sha256)

                    // This optional output is only a 90-array payload, never a model directory.
                    // It is unreachable until all 30 original-source Q4 comparisons pass.
                    if let q8Path = environment["MIDNIGHT_GEMMA4_ROUTER_Q8_OUTPUT"] {
                        let q8URL = URL(fileURLWithPath: q8Path)
                        try #require(q8URL.pathExtension == "safetensors")
                        try #require(!FileManager.default.fileExists(atPath: q8URL.path))
                        let identity = try writeQ8(source, to: q8URL)
                        report["q8_tensor_payload"] = identity
                        report["q8_tensor_payload_created"] = true
                    }
                }
                try verifyInstalled(snapshot, directory: URL(fileURLWithPath: manifest.installed))
                try #require(hash(try Data(contentsOf: manifestURL)) == hash(manifestData))
                report["status"] = "passed"
                report["installed_selected_tensors_unchanged"] = true
                try writeReport(report, to: output)
            } catch {
                report["status"] = "failed"
                report["error"] = String(describing: error)
                try writeReport(report, to: output)
                throw error
            }
        }

        private func writeQ8(_ source: [String: MLXArray], to url: URL) throws -> [String: Any] {
            var arrays: [String: MLXArray] = [:]
            for layer in 0..<30 {
                let prefix = "language_model.model.layers.\(layer).router.proj"
                let original = try array(source, prefix + ".weight", dtype: .bfloat16, shape: [128, 2816])
                let result = MLX.quantized(original, groupSize: 64, bits: 8, mode: .affine)
                let biases = try #require(result.biases)
                eval(result.wq, result.scales, biases)
                try #require(result.wq.dtype == .uint32 && result.wq.shape == [128, 704])
                for metadata in [result.scales, biases] {
                    try #require(metadata.dtype == .bfloat16 && metadata.shape == [128, 44])
                    try #require(all(isFinite(metadata)).item(Bool.self))
                }
                arrays[prefix + ".weight"] = result.wq
                arrays[prefix + ".scales"] = result.scales
                arrays[prefix + ".biases"] = biases
            }
            try #require(arrays.count == 90)
            let payloadBytes = arrays.values.reduce(0) { $0 + bytes($1).count }
            try #require(payloadBytes == 11_489_280)
            try MLX.save(arrays: arrays, url: url)
            let reloaded = try loadArrays(url: url)
            try #require(Set(reloaded.keys) == Set(arrays.keys))
            for (name, value) in arrays {
                let saved = try #require(reloaded[name])
                try #require(saved.dtype == value.dtype && saved.shape == value.shape && bytes(saved) == bytes(value))
            }
            return [
                "path": url.path, "sha256": hash(try Data(contentsOf: url)), "tensor_count": 90,
                "payload_bytes": payloadBytes, "bits": 8, "group_size": 64, "metadata_dtype": "BF16",
            ]
        }

        private func verifyKeys(_ source: [String: MLXArray], baseline: [String: MLXArray]) throws {
            let sourceKeys = Set(
                (0..<30).flatMap { layer in
                    ["proj.weight", "scale", "per_expert_scale"].map {
                        "language_model.model.layers.\(layer).router.\($0)"
                    }
                })
            let baselineKeys = Set(
                (0..<30).flatMap { layer in
                    ["proj.weight", "proj.scales", "proj.biases", "scale", "per_expert_scale"].map {
                        "language_model.model.layers.\(layer).router.\($0)"
                    }
                })
            try #require(Set(source.keys) == sourceKeys && Set(baseline.keys) == baselineKeys)
        }

        private func array(_ arrays: [String: MLXArray], _ name: String, dtype: DType, shape: [Int]) throws -> MLXArray
        {
            let value = try #require(arrays[name])
            try #require(value.dtype == dtype && value.shape == shape)
            return value
        }

        private func verifiedFile(_ directory: URL, _ identity: ArrayFile, expectedName: String) throws -> URL {
            try #require(identity.path == expectedName && identity.bytes < 24 * 1024 * 1024)
            let url = directory.appendingPathComponent(identity.path)
            let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize
            try #require(size == identity.bytes)
            let data = try Data(contentsOf: url)
            try #require(data.count == identity.bytes && hash(data) == identity.sha256)
            return url
        }

        private func verifyInstalled(_ snapshot: InstalledSnapshot, directory: URL) throws {
            try #require(snapshot.metadata.count == 3 && snapshot.tensors.count == 150)
            for (name, expected) in snapshot.metadata {
                try #require(
                    ["config.json", "model.safetensors.index.json", "scale-search-quantization.json"].contains(name))
                try #require(hash(try Data(contentsOf: directory.appendingPathComponent(name))) == expected)
            }
            for tensor in snapshot.tensors {
                try #require(tensor.file.hasPrefix("model-") && tensor.file.hasSuffix(".safetensors"))
                try #require(!tensor.file.contains("/") && tensor.absoluteRange.count == 2)
                let start = tensor.absoluteRange[0]
                let end = tensor.absoluteRange[1]
                try #require(start >= 8 && end >= start && end - start + 1 == tensor.bytes)
                try #require(tensor.bytes > 0 && tensor.bytes < 1024 * 1024)
                let handle = try FileHandle(forReadingFrom: directory.appendingPathComponent(tensor.file))
                defer { try? handle.close() }
                try handle.seek(toOffset: UInt64(start))
                let data = try #require(try handle.read(upToCount: tensor.bytes))
                try #require(data.count == tensor.bytes && hash(data) == tensor.sha256)
            }
        }

        private func bytes(_ array: MLXArray) -> Data { array.asData(access: .copy).data }

        private func hash(_ data: Data) -> String {
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }

        private func writeReport(_ value: [String: Any], to url: URL, initial: Bool = false) throws {
            let data = try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: url, options: initial ? .withoutOverwriting : .atomic)
        }

        private struct ArrayFile: Decodable {
            var path: String
            var bytes: Int
            var sha256: String
            var tensorCount: Int
            var payloadBytes: Int
        }

        private struct Recovery: Decodable {
            var status: String
            var revision: String
            var routerCount: Int
            var installed: String
            var sourceCompanionsExact: Bool
            var installedSelectedTensorsUnchanged: Bool
            var candidateCreated: Bool
            var sourceArrays: ArrayFile
            var baselineArrays: ArrayFile
        }

        private struct InstalledSnapshot: Decodable {
            var metadata: [String: String]
            var tensors: [InstalledTensor]
        }

        private struct InstalledTensor: Decodable {
            var file: String
            var absoluteRange: [Int]
            var bytes: Int
            var sha256: String
        }
    }
#endif
