import ArgumentParser
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import ModelQualityCore
import ModelRunnerCore
import ModelRunnerProtocol
import Tokenizers

private struct TeacherScoringPayload: Sendable {
    var samples: [ModelQualityCorpusSample]
    var maximumTokensPerSample: Int
    var cacheDirectory: URL
}

private struct StudentScoringPayload: Sendable {
    var teacherSamples: [TeacherCacheSample]
    var positionChunkSize: Int
    var cacheDirectory: URL
}

private struct TeacherCacheSample: Codable, Sendable {
    var quality: ModelQualitySampleResult
    var text: String
    var tokenIDs: [Int]
    var logitsFile: String
    var logitsShape: [Int]
    var logitsBytes: Int
    var groundTruthTop1CorrectCount: Int

    enum CodingKeys: String, CodingKey {
        case quality, text
        case tokenIDs = "token_ids"
        case logitsFile = "logits_file"
        case logitsShape = "logits_shape"
        case logitsBytes = "logits_bytes"
        case groundTruthTop1CorrectCount = "ground_truth_top1_correct_count"
    }
}

private struct TeacherCacheManifest: Codable, Sendable {
    var format = 1
    var strategy = "deterministic_two_pass_full_vocab_logits"
    var logitsDType = "float32"
    var teacherPath: String
    var teacherCheckpointFingerprint: String
    var corpusFingerprint: String
    var tokenIDFingerprint: String
    var samples: [TeacherCacheSample]

    enum CodingKeys: String, CodingKey {
        case format, strategy, samples
        case logitsDType = "logits_dtype"
        case teacherPath = "teacher_path"
        case teacherCheckpointFingerprint = "teacher_checkpoint_fingerprint"
        case corpusFingerprint = "corpus_fingerprint"
        case tokenIDFingerprint = "token_id_fingerprint"
    }
}

private struct TeacherSampleReport: Encodable, Sendable {
    var id: String
    var category: String?
    var originalTokenCount: Int
    var evaluatedTokenCount: Int
    var scoredTokenCount: Int
    var truncated: Bool
    var tokenIDFingerprint: String
    var nllSum: Double
    var nll: Double
    var perplexity: Double
    var groundTruthTop1CorrectCount: Int
    var groundTruthTop1Accuracy: Double

    init(_ sample: TeacherCacheSample) {
        let quality = sample.quality
        id = quality.id
        category = quality.category
        originalTokenCount = quality.originalTokenCount
        evaluatedTokenCount = quality.evaluatedTokenCount
        scoredTokenCount = quality.scoredTokenCount
        truncated = quality.truncated
        tokenIDFingerprint = quality.tokenIDFingerprint
        nllSum = quality.nllSum
        nll = quality.nll
        perplexity = quality.perplexity
        groundTruthTop1CorrectCount = sample.groundTruthTop1CorrectCount
        groundTruthTop1Accuracy =
            Double(sample.groundTruthTop1CorrectCount) / Double(quality.scoredTokenCount)
    }

    enum CodingKeys: String, CodingKey {
        case id, category, truncated, nll, perplexity
        case originalTokenCount = "original_token_count"
        case evaluatedTokenCount = "evaluated_token_count"
        case scoredTokenCount = "scored_token_count"
        case tokenIDFingerprint = "token_id_fingerprint"
        case nllSum = "nll_sum"
        case groundTruthTop1CorrectCount = "ground_truth_top1_correct_count"
        case groundTruthTop1Accuracy = "ground_truth_top1_accuracy"
    }
}

private struct TeacherPhaseReport: Encodable, Sendable {
    var modelPath: String
    var modelType: String?
    var declaredDType: String?
    var checkpointFingerprint: String
    var checkpointFingerprintMethod: String
    var elapsedSeconds: Double
    var mlxPeakMemoryBytes: Int
    var summary: ModelQualitySummary
    var groundTruthTop1CorrectCount: Int
    var groundTruthTop1Accuracy: Double
    var samples: [TeacherSampleReport]

    enum CodingKeys: String, CodingKey {
        case summary, samples
        case modelPath = "model_path"
        case modelType = "model_type"
        case declaredDType = "declared_dtype"
        case checkpointFingerprint = "checkpoint_fingerprint"
        case checkpointFingerprintMethod = "checkpoint_fingerprint_method"
        case elapsedSeconds = "elapsed_seconds"
        case mlxPeakMemoryBytes = "mlx_peak_memory_bytes"
        case groundTruthTop1CorrectCount = "ground_truth_top1_correct_count"
        case groundTruthTop1Accuracy = "ground_truth_top1_accuracy"
    }
}

private struct TeacherPassResult: Sendable {
    var samples: [TeacherCacheSample]
    var tokenIDFingerprint: String
    var elapsedSeconds: Double
    var mlxPeakMemoryBytes: Int
}

private struct PairedSampleResult: Encodable, Sendable {
    var id: String
    var category: String?
    var scoredTokenCount: Int
    var tokenIDFingerprint: String
    var teacherNLL: Double
    var studentNLL: Double
    var studentMinusTeacherNLL: Double
    var teacherKLSum: Double
    var teacherKL: Double
    var teacherStudentTop1AgreementCount: Int
    var teacherStudentTop1Agreement: Double
    var teacherGroundTruthTop1CorrectCount: Int
    var studentGroundTruthTop1CorrectCount: Int
    var teacherCorrectStudentWrongCount: Int
    var teacherWrongStudentCorrectCount: Int

    enum CodingKeys: String, CodingKey {
        case id, category
        case scoredTokenCount = "scored_token_count"
        case tokenIDFingerprint = "token_id_fingerprint"
        case teacherNLL = "teacher_nll"
        case studentNLL = "student_nll"
        case studentMinusTeacherNLL = "student_minus_teacher_nll"
        case teacherKLSum = "teacher_kl_sum"
        case teacherKL = "teacher_kl"
        case teacherStudentTop1AgreementCount = "teacher_student_top1_agreement_count"
        case teacherStudentTop1Agreement = "teacher_student_top1_agreement"
        case teacherGroundTruthTop1CorrectCount = "teacher_ground_truth_top1_correct_count"
        case studentGroundTruthTop1CorrectCount = "student_ground_truth_top1_correct_count"
        case teacherCorrectStudentWrongCount = "teacher_correct_student_wrong_count"
        case teacherWrongStudentCorrectCount = "teacher_wrong_student_correct_count"
    }
}

private struct StudentReport: Encodable, Sendable {
    var modelPath: String
    var modelType: String?
    var declaredDType: String?
    var checkpointFingerprint: String
    var checkpointFingerprintMethod: String
    var tokenIDFingerprint: String
    var elapsedSeconds: Double
    var mlxPeakMemoryBytes: Int
    var summary: ModelQualitySummary
    var studentMinusTeacherNLL: Double
    var teacherKLSum: Double
    var tokenWeightedTeacherKL: Double
    var teacherStudentTop1AgreementCount: Int
    var teacherStudentTop1Agreement: Double
    var studentGroundTruthTop1CorrectCount: Int
    var studentGroundTruthTop1Accuracy: Double
    var teacherCorrectStudentWrongCount: Int
    var teacherWrongStudentCorrectCount: Int
    var samples: [PairedSampleResult]

    enum CodingKeys: String, CodingKey {
        case summary, samples
        case modelPath = "model_path"
        case modelType = "model_type"
        case declaredDType = "declared_dtype"
        case checkpointFingerprint = "checkpoint_fingerprint"
        case checkpointFingerprintMethod = "checkpoint_fingerprint_method"
        case tokenIDFingerprint = "token_id_fingerprint"
        case elapsedSeconds = "elapsed_seconds"
        case mlxPeakMemoryBytes = "mlx_peak_memory_bytes"
        case studentMinusTeacherNLL = "student_minus_teacher_nll"
        case teacherKLSum = "teacher_kl_sum"
        case tokenWeightedTeacherKL = "token_weighted_teacher_kl"
        case teacherStudentTop1AgreementCount = "teacher_student_top1_agreement_count"
        case teacherStudentTop1Agreement = "teacher_student_top1_agreement"
        case studentGroundTruthTop1CorrectCount = "student_ground_truth_top1_correct_count"
        case studentGroundTruthTop1Accuracy = "student_ground_truth_top1_accuracy"
        case teacherCorrectStudentWrongCount = "teacher_correct_student_wrong_count"
        case teacherWrongStudentCorrectCount = "teacher_wrong_student_correct_count"
    }
}

private struct StudentPassMeasurements: Sendable {
    var qualitySamples: [ModelQualitySampleResult]
    var pairedSamples: [PairedSampleResult]
    var tokenIDFingerprint: String
    var teacherKLSum: Double
    var teacherStudentTop1AgreementCount: Int
    var studentGroundTruthTop1CorrectCount: Int
    var teacherCorrectStudentWrongCount: Int
    var teacherWrongStudentCorrectCount: Int
}

private struct TeacherLogitCacheReport: Encodable, Sendable {
    var strategy = "deterministic_two_pass_full_vocab_logits"
    var logitsDType = "float32"
    var probabilityAndKLComputeDType = "float32"
    var fullVocabulary = true
    var cacheBytes: Int
    var fingerprint: String
    var retained: Bool
    var path: String?

    enum CodingKeys: String, CodingKey {
        case strategy, retained, path
        case logitsDType = "logits_dtype"
        case probabilityAndKLComputeDType = "probability_and_kl_compute_dtype"
        case fullVocabulary = "full_vocabulary"
        case cacheBytes = "cache_bytes"
        case fingerprint
    }
}

private struct TeacherKLBenchmarkReport: Encodable, Sendable {
    var format = 1
    var status = "measured"
    var metric = "teacher_forced_full_vocab_kl_teacher_to_student"
    var klDirection = "KL(teacher || student)"
    var referenceKind: String
    var representsBF16Teacher: Bool
    var createdAt: String
    var corpusPath: String
    var corpusFingerprint: String
    var tokenIDFingerprint: String
    var backend: String
    var device: String
    var addSpecialTokens = true
    var maximumTokensPerSample: Int
    var positionChunkSize: Int
    var sampleCount: Int
    var scoredTokenCount: Int
    var elapsedSeconds: Double
    var teacherLogitCache: TeacherLogitCacheReport
    var teacher: TeacherPhaseReport
    var students: [StudentReport]

    enum CodingKeys: String, CodingKey {
        case format, status, metric, backend, device, teacher, students
        case klDirection = "kl_direction"
        case referenceKind = "reference_kind"
        case representsBF16Teacher = "represents_bf16_teacher"
        case createdAt = "created_at"
        case corpusPath = "corpus_path"
        case corpusFingerprint = "corpus_fingerprint"
        case tokenIDFingerprint = "token_id_fingerprint"
        case addSpecialTokens = "add_special_tokens"
        case maximumTokensPerSample = "maximum_tokens_per_sample"
        case positionChunkSize = "position_chunk_size"
        case sampleCount = "sample_count"
        case scoredTokenCount = "scored_token_count"
        case elapsedSeconds = "elapsed_seconds"
        case teacherLogitCache = "teacher_logit_cache"
    }
}

private enum TeacherKLBenchmarkError: Error, LocalizedError {
    case invalidInput(String)
    case insufficientTokens(sampleID: String, count: Int)
    case tokenizerMismatch(sampleID: String, modelPath: String)
    case logitsShapeMismatch(sampleID: String, teacher: [Int], student: [Int])
    case invalidMeasurement(sampleID: String, detail: String)

    var errorDescription: String? {
        switch self {
        case .invalidInput(let detail):
            "Invalid teacher KL benchmark input: \(detail)"
        case .insufficientTokens(let sampleID, let count):
            "Quality sample '\(sampleID)' encoded to \(count) token(s); at least 2 are required."
        case .tokenizerMismatch(let sampleID, let modelPath):
            "Student tokenizer at \(modelPath) disagrees with the teacher on sample '\(sampleID)'."
        case .logitsShapeMismatch(let sampleID, let teacher, let student):
            "Logits shape mismatch for '\(sampleID)': teacher \(teacher), student \(student)."
        case .invalidMeasurement(let sampleID, let detail):
            "Invalid teacher KL measurement for '\(sampleID)': \(detail)"
        }
    }
}

@main
private struct ModelTeacherKLBenchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "model-runner-teacher-kl-bench",
        abstract:
            "Compare local MLX students with a BF16 teacher using exact full-vocabulary FP32 KL."
    )

    @Argument(help: "Local BF16 teacher checkpoint directory.")
    var teacher: String

    @Argument(help: "Deterministic JSONL corpus path.")
    var corpus: String

    @Argument(help: "JSON report path.")
    var output: String

    @Option(
        name: .customLong("student"),
        help: "Local student checkpoint directory; repeat for every candidate."
    )
    var students: [String] = []

    @Option(help: "Maximum encoded tokens evaluated per sample (2...2048).")
    var maxTokensPerSample = 512

    @Option(help: "Token-position chunk size for the full-vocabulary FP32 KL reduction.")
    var positionChunkSize = 16

    @Option(
        help:
            "New directory for temporary FP32 teacher logits; defaults to a unique system temp directory."
    )
    var teacherCacheDirectory: String?

    @Flag(help: "Retain the teacher-logit cache and its manifest after writing the report.")
    var keepTeacherCache = false

    @Flag(help: "Explicitly allow a quantized reference; reports mark its fidelity provisional rather than BF16.")
    var allowQuantizedReference = false

    @Flag(help: "Run on CPU instead of the default MLX device.")
    var cpu = false

    @Flag(help: "Replace an existing report file.")
    var overwrite = false

    mutating func validate() throws {
        guard !students.isEmpty else {
            throw ValidationError("Pass at least one --student checkpoint.")
        }
        guard (2...2_048).contains(maxTokensPerSample) else {
            throw ValidationError("--max-tokens-per-sample must be in 2...2048.")
        }
        guard (1...2_048).contains(positionChunkSize) else {
            throw ValidationError("--position-chunk-size must be in 1...2048.")
        }
        let normalizedStudents = students.map {
            NSString(string: $0).expandingTildeInPath
        }
        guard Set(normalizedStudents).count == normalizedStudents.count else {
            throw ValidationError("--student checkpoint paths must be unique.")
        }
    }

    mutating func run() async throws {
        let teacherURL = localURL(teacher, isDirectory: true)
        let studentURLs = students.map { localURL($0, isDirectory: true) }
        let corpusURL = localURL(corpus)
        let outputURL = localURL(output)
        try validateInputs(
            teacherURL: teacherURL,
            studentURLs: studentURLs,
            corpusURL: corpusURL,
            outputURL: outputURL
        )

        let samples = try ModelQualityCore.loadCorpus(from: corpusURL)
        let corpusFingerprint = ModelQualityCore.corpusFingerprint(samples)
        let teacherFingerprint = try checkpointFingerprint(teacherURL)
        let cacheURL = try createTeacherCacheDirectory()
        let shouldRemoveCache = !keepTeacherCache
        defer {
            if shouldRemoveCache {
                try? FileManager.default.removeItem(at: cacheURL)
            }
        }

        let totalStartedAt = ContinuousClock.now
        if checkpointIsQuantized(teacherURL) {
            print("Quantized reference: fidelity is provisional and does not measure distance from BF16.")
        }
        print("Teacher pass: \(teacherURL.path)")
        let teacherPass = try await scoreTeacher(
            modelURL: teacherURL,
            samples: samples,
            cacheDirectory: cacheURL
        )
        Memory.clearCache()

        let teacherQualitySamples = teacherPass.samples.map(\.quality)
        let teacherSummary = try ModelQualityCore.summarize(teacherQualitySamples)
        let teacherReport = makeTeacherReport(
            teacherURL: teacherURL, fingerprint: teacherFingerprint,
            teacherPass: teacherPass, summary: teacherSummary
        )

        let (cacheBytes, cacheFingerprint) = try writeCacheManifest(
            teacherURL: teacherURL, teacherFingerprint: teacherFingerprint,
            corpusFingerprint: corpusFingerprint, teacherPass: teacherPass, cacheURL: cacheURL
        )

        let studentReports = try await scoreStudents(
            studentURLs,
            teacherSamples: teacherPass.samples,
            teacherSummary: teacherSummary,
            cacheDirectory: cacheURL
        )

        let report = makeBenchmarkReport(
            corpusURL: corpusURL, corpusFingerprint: corpusFingerprint,
            teacherPass: teacherPass, teacherSummary: teacherSummary,
            teacherReport: teacherReport, studentReports: studentReports,
            cacheBytes: cacheBytes, cacheFingerprint: cacheFingerprint,
            cacheURL: cacheURL, startedAt: totalStartedAt
        )

        try writeReport(
            report, studentReports: studentReports, outputURL: outputURL, cacheURL: cacheURL
        )
    }

    private func makeTeacherReport(
        teacherURL: URL, fingerprint: String, teacherPass: TeacherPassResult,
        summary: ModelQualitySummary
    ) -> TeacherPhaseReport {
        let teacherCorrect = teacherPass.samples.reduce(0) {
            $0 + $1.groundTruthTop1CorrectCount
        }
        return TeacherPhaseReport(
            modelPath: teacherURL.path,
            modelType: checkpointModelType(teacherURL),
            declaredDType: checkpointDeclaredDType(teacherURL),
            checkpointFingerprint: fingerprint,
            checkpointFingerprintMethod: checkpointFingerprintMethod,
            elapsedSeconds: teacherPass.elapsedSeconds,
            mlxPeakMemoryBytes: teacherPass.mlxPeakMemoryBytes,
            summary: summary,
            groundTruthTop1CorrectCount: teacherCorrect,
            groundTruthTop1Accuracy: Double(teacherCorrect) / Double(summary.scoredTokenCount),
            samples: teacherPass.samples.map(TeacherSampleReport.init)
        )
    }

    private func makeBenchmarkReport(
        corpusURL: URL, corpusFingerprint: String, teacherPass: TeacherPassResult,
        teacherSummary: ModelQualitySummary, teacherReport: TeacherPhaseReport,
        studentReports: [StudentReport], cacheBytes: Int, cacheFingerprint: String,
        cacheURL: URL, startedAt: ContinuousClock.Instant
    ) -> TeacherKLBenchmarkReport {
        let cacheReport = TeacherLogitCacheReport(
            cacheBytes: cacheBytes,
            fingerprint: cacheFingerprint,
            retained: keepTeacherCache,
            path: keepTeacherCache ? cacheURL.path : nil
        )
        return TeacherKLBenchmarkReport(
            referenceKind: checkpointIsQuantized(URL(fileURLWithPath: teacherReport.modelPath))
                ? "quantized_reference_provisional" : "unquantized_bf16_teacher",
            representsBF16Teacher: !checkpointIsQuantized(URL(fileURLWithPath: teacherReport.modelPath)),
            createdAt: ISO8601DateFormatter().string(from: Date()),
            corpusPath: corpusURL.path,
            corpusFingerprint: corpusFingerprint,
            tokenIDFingerprint: teacherPass.tokenIDFingerprint,
            backend: backendName,
            device: cpu ? "cpu" : (Device.defaultDevice().deviceType?.rawValue ?? "unknown"),
            maximumTokensPerSample: maxTokensPerSample,
            positionChunkSize: positionChunkSize,
            sampleCount: teacherSummary.sampleCount,
            scoredTokenCount: teacherSummary.scoredTokenCount,
            elapsedSeconds: seconds(startedAt.duration(to: .now)),
            teacherLogitCache: cacheReport,
            teacher: teacherReport,
            students: studentReports
        )
    }

    private func writeReport(
        _ report: TeacherKLBenchmarkReport, studentReports: [StudentReport],
        outputURL: URL, cacheURL: URL
    ) throws {
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try writeJSON(report, to: outputURL)
        for student in studentReports {
            print(
                String(
                    format: "%@: KL %.8f, NLL %.6f (%+.6f), teacher top-1 agreement %.2f%%",
                    URL(fileURLWithPath: student.modelPath).lastPathComponent,
                    student.tokenWeightedTeacherKL,
                    student.summary.tokenWeightedNLL,
                    student.studentMinusTeacherNLL,
                    student.teacherStudentTop1Agreement * 100
                )
            )
        }
        print("Wrote \(outputURL.path)")
        if keepTeacherCache {
            print("Retained teacher cache at \(cacheURL.path)")
        }
    }

    private func writeCacheManifest(
        teacherURL: URL, teacherFingerprint: String, corpusFingerprint: String,
        teacherPass: TeacherPassResult, cacheURL: URL
    ) throws -> (Int, String) {
        let manifest = TeacherCacheManifest(
            teacherPath: teacherURL.path,
            teacherCheckpointFingerprint: teacherFingerprint,
            corpusFingerprint: corpusFingerprint,
            tokenIDFingerprint: teacherPass.tokenIDFingerprint,
            samples: teacherPass.samples
        )
        let manifestURL = cacheURL.appendingPathComponent("manifest.json")
        try writeJSON(manifest, to: manifestURL)
        let cacheBytes = try directoryByteCount(cacheURL)
        let cacheFingerprint = teacherCacheFingerprint(manifest)

        return (cacheBytes, cacheFingerprint)
    }

    private func scoreStudents(
        _ studentURLs: [URL],
        teacherSamples: [TeacherCacheSample],
        teacherSummary: ModelQualitySummary,
        cacheDirectory: URL
    ) async throws -> [StudentReport] {
        var studentReports = [StudentReport]()
        studentReports.reserveCapacity(studentURLs.count)
        for (index, studentURL) in studentURLs.enumerated() {
            print("Student pass \(index + 1)/\(studentURLs.count): \(studentURL.path)")
            let studentFingerprint = try checkpointFingerprint(studentURL)
            let startedAt = ContinuousClock.now
            Memory.peakMemory = 0
            let measurements = try await scoreStudent(
                modelURL: studentURL,
                teacherSamples: teacherSamples,
                cacheDirectory: cacheDirectory
            )
            let peakMemory = Memory.peakMemory
            let elapsed = seconds(startedAt.duration(to: .now))
            let summary = try ModelQualityCore.summarize(measurements.qualitySamples)
            guard summary.scoredTokenCount == teacherSummary.scoredTokenCount else {
                throw TeacherKLBenchmarkError.invalidMeasurement(
                    sampleID: "aggregate",
                    detail: "teacher and student scored-token counts differ"
                )
            }
            let count = summary.scoredTokenCount
            studentReports.append(
                StudentReport(
                    modelPath: studentURL.path,
                    modelType: checkpointModelType(studentURL),
                    declaredDType: checkpointDeclaredDType(studentURL),
                    checkpointFingerprint: studentFingerprint,
                    checkpointFingerprintMethod: checkpointFingerprintMethod,
                    tokenIDFingerprint: measurements.tokenIDFingerprint,
                    elapsedSeconds: elapsed,
                    mlxPeakMemoryBytes: peakMemory,
                    summary: summary,
                    studentMinusTeacherNLL: summary.tokenWeightedNLL - teacherSummary.tokenWeightedNLL,
                    teacherKLSum: measurements.teacherKLSum,
                    tokenWeightedTeacherKL: measurements.teacherKLSum / Double(count),
                    teacherStudentTop1AgreementCount: measurements.teacherStudentTop1AgreementCount,
                    teacherStudentTop1Agreement:
                        Double(measurements.teacherStudentTop1AgreementCount) / Double(count),
                    studentGroundTruthTop1CorrectCount: measurements.studentGroundTruthTop1CorrectCount,
                    studentGroundTruthTop1Accuracy:
                        Double(measurements.studentGroundTruthTop1CorrectCount) / Double(count),
                    teacherCorrectStudentWrongCount: measurements.teacherCorrectStudentWrongCount,
                    teacherWrongStudentCorrectCount: measurements.teacherWrongStudentCorrectCount,
                    samples: measurements.pairedSamples
                )
            )
            Memory.clearCache()
        }
        return studentReports
    }

    private func scoreTeacher(
        modelURL: URL,
        samples: [ModelQualityCorpusSample],
        cacheDirectory: URL
    ) async throws -> TeacherPassResult {
        let payload = TeacherScoringPayload(
            samples: samples,
            maximumTokensPerSample: maxTokensPerSample,
            cacheDirectory: cacheDirectory
        )
        let resourceLimits = try MLXResourceGuard.resolve(
            for: cpu ? .cpu : benchmarkEngine,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
        )
        let startedAt = ContinuousClock.now

        let operation: @Sendable () async throws -> TeacherPassResult = {
            Memory.peakMemory = 0
            await TalkieModelRegistration.register()
            try MLXResourceGuard.apply(resourceLimits)
            let container = try await #huggingFaceLoadModelContainer(
                configuration: ModelConfiguration(directory: modelURL)
            )
            let result = try await container.perform(values: payload) { context, payload in
                context.model.train(false)
                var cachedSamples = [TeacherCacheSample]()
                var tokenSequences = [ModelQualityTokenSequence]()
                cachedSamples.reserveCapacity(payload.samples.count)
                tokenSequences.reserveCapacity(payload.samples.count)

                for (index, sample) in payload.samples.enumerated() {
                    let originalTokens = context.tokenizer.encode(
                        text: sample.text,
                        addSpecialTokens: true
                    )
                    let tokens = try ModelQualityCore.boundedTokens(
                        originalTokens,
                        maximumCount: payload.maximumTokensPerSample
                    )
                    guard tokens.count >= 2 else {
                        throw TeacherKLBenchmarkError.insufficientTokens(
                            sampleID: sample.id,
                            count: tokens.count
                        )
                    }

                    let inputCount = tokens.count - 1
                    let inputs = MLXArray(Array(tokens.dropLast())).reshaped(1, inputCount)
                    let targets = MLXArray(Array(tokens.dropFirst())).reshaped(1, inputCount)
                    let logits = context.model(inputs, cache: nil).asType(.float32)
                    let lossSum = MLXFast.crossEntropy(logits: logits, targets: targets).sum()
                    let correct = (logits.argMax(axis: -1) .== targets).asType(.int32).sum()
                    MLX.eval(logits, lossSum, correct)

                    let logitsFile = String(format: "sample-%05d.npy", index)
                    let logitsURL = payload.cacheDirectory.appendingPathComponent(logitsFile)
                    try save(array: logits, url: logitsURL)
                    let logitsBytes = try fileByteCount(logitsURL)
                    let tokenFingerprint = ModelQualityCore.tokenIDFingerprint(tokens)
                    let quality = try ModelQualitySampleResult(
                        id: sample.id,
                        category: sample.category,
                        originalTokenCount: originalTokens.count,
                        evaluatedTokenCount: tokens.count,
                        tokenIDFingerprint: tokenFingerprint,
                        nllSum: Double(lossSum.item(Float.self))
                    )
                    let top1Correct = Int(correct.item(Int32.self))
                    cachedSamples.append(
                        TeacherCacheSample(
                            quality: quality,
                            text: sample.text,
                            tokenIDs: tokens,
                            logitsFile: logitsFile,
                            logitsShape: logits.shape,
                            logitsBytes: logitsBytes,
                            groundTruthTop1CorrectCount: top1Correct
                        )
                    )
                    tokenSequences.append(
                        ModelQualityTokenSequence(sampleID: sample.id, tokenIDs: tokens)
                    )
                    Memory.clearCache()
                    print(
                        "teacher sample \(index + 1)/\(payload.samples.count) \(sample.id): "
                            + String(format: "NLL %.6f, cache %.1f MiB", quality.nll, mib(logitsBytes))
                    )
                }

                return TeacherPassResult(
                    samples: cachedSamples,
                    tokenIDFingerprint: ModelQualityCore.combinedTokenIDFingerprint(tokenSequences),
                    elapsedSeconds: 0,
                    mlxPeakMemoryBytes: 0
                )
            }
            let peakMemory = Memory.peakMemory
            return TeacherPassResult(
                samples: result.samples,
                tokenIDFingerprint: result.tokenIDFingerprint,
                elapsedSeconds: seconds(startedAt.duration(to: .now)),
                mlxPeakMemoryBytes: peakMemory
            )
        }

        if cpu {
            return try await Device.withDefaultDevice(.cpu, operation)
        }
        return try await operation()
    }

    private func scoreStudent(
        modelURL: URL,
        teacherSamples: [TeacherCacheSample],
        cacheDirectory: URL
    ) async throws -> StudentPassMeasurements {
        let payload = StudentScoringPayload(
            teacherSamples: teacherSamples,
            positionChunkSize: positionChunkSize,
            cacheDirectory: cacheDirectory
        )
        let resourceLimits = try MLXResourceGuard.resolve(
            for: cpu ? .cpu : benchmarkEngine,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
        )

        let operation: @Sendable () async throws -> StudentPassMeasurements = {
            await TalkieModelRegistration.register()
            try MLXResourceGuard.apply(resourceLimits)
            let container = try await #huggingFaceLoadModelContainer(
                configuration: ModelConfiguration(directory: modelURL)
            )
            return try await container.perform(values: payload) { context, payload in
                context.model.train(false)
                var qualitySamples = [ModelQualitySampleResult]()
                var pairedSamples = [PairedSampleResult]()
                var tokenSequences = [ModelQualityTokenSequence]()
                var aggregateKLSum = 0.0
                var aggregateAgreement = 0
                var aggregateStudentCorrect = 0
                var aggregateRegressions = 0
                var aggregateImprovements = 0
                qualitySamples.reserveCapacity(payload.teacherSamples.count)
                pairedSamples.reserveCapacity(payload.teacherSamples.count)
                tokenSequences.reserveCapacity(payload.teacherSamples.count)

                for (index, teacherSample) in payload.teacherSamples.enumerated() {
                    let quality = teacherSample.quality
                    let (studentQuality, paired) = try Self.scoreStudentSample(
                        context: context, teacherSample: teacherSample, modelURL: modelURL,
                        cacheDirectory: payload.cacheDirectory, positionChunkSize: payload.positionChunkSize
                    )
                    qualitySamples.append(studentQuality)
                    pairedSamples.append(paired)
                    tokenSequences.append(
                        ModelQualityTokenSequence(sampleID: quality.id, tokenIDs: teacherSample.tokenIDs)
                    )
                    aggregateKLSum += paired.teacherKLSum
                    aggregateAgreement += paired.teacherStudentTop1AgreementCount
                    aggregateStudentCorrect += paired.studentGroundTruthTop1CorrectCount
                    aggregateRegressions += paired.teacherCorrectStudentWrongCount
                    aggregateImprovements += paired.teacherWrongStudentCorrectCount
                    Memory.clearCache()
                    print(
                        "student sample \(index + 1)/\(payload.teacherSamples.count) \(quality.id): "
                            + String(
                                format: "KL %.8f, NLL %.6f, agreement %.2f%%",
                                paired.teacherKL,
                                paired.studentNLL,
                                paired.teacherStudentTop1Agreement * 100
                            )
                    )
                }

                return StudentPassMeasurements(
                    qualitySamples: qualitySamples,
                    pairedSamples: pairedSamples,
                    tokenIDFingerprint: ModelQualityCore.combinedTokenIDFingerprint(tokenSequences),
                    teacherKLSum: aggregateKLSum,
                    teacherStudentTop1AgreementCount: aggregateAgreement,
                    studentGroundTruthTop1CorrectCount: aggregateStudentCorrect,
                    teacherCorrectStudentWrongCount: aggregateRegressions,
                    teacherWrongStudentCorrectCount: aggregateImprovements
                )
            }
        }

        if cpu {
            return try await Device.withDefaultDevice(.cpu, operation)
        }
        return try await operation()
    }

    private static func scoreStudentSample(
        context: ModelContext, teacherSample: TeacherCacheSample, modelURL: URL,
        cacheDirectory: URL, positionChunkSize: Int
    ) throws -> (ModelQualitySampleResult, PairedSampleResult) {
        let quality = teacherSample.quality
        let studentTokens = context.tokenizer.encode(
            text: teacherSample.text,
            addSpecialTokens: true
        )
        let boundedStudentTokens = try ModelQualityCore.boundedTokens(
            studentTokens,
            maximumCount: teacherSample.tokenIDs.count
        )
        guard studentTokens.count == quality.originalTokenCount,
            boundedStudentTokens == teacherSample.tokenIDs
        else {
            throw TeacherKLBenchmarkError.tokenizerMismatch(
                sampleID: quality.id,
                modelPath: modelURL.path
            )
        }

        let inputCount = teacherSample.tokenIDs.count - 1
        let inputs = MLXArray(Array(teacherSample.tokenIDs.dropLast())).reshaped(1, inputCount)
        let targets = MLXArray(Array(teacherSample.tokenIDs.dropFirst())).reshaped(1, inputCount)
        let studentLogits = context.model(inputs, cache: nil).asType(.float32)
        let teacherLogitsURL = cacheDirectory.appendingPathComponent(
            teacherSample.logitsFile
        )
        guard try fileByteCount(teacherLogitsURL) == teacherSample.logitsBytes else {
            throw TeacherKLBenchmarkError.invalidMeasurement(
                sampleID: quality.id,
                detail: "cached teacher-logit file size changed between passes"
            )
        }
        // MLX's `.npy` Load primitive is CPU-only on the Metal backend. Materialize
        // the FP32 cache entry there before default-stream GPU operations consume it;
        // Apple silicon unified memory makes the evaluated buffer available without
        // a dtype conversion or an explicit device copy.
        let teacherLogits = try loadArray(url: teacherLogitsURL, stream: .cpu)
        MLX.eval(teacherLogits)
        guard teacherLogits.shape == teacherSample.logitsShape,
            studentLogits.shape == teacherSample.logitsShape
        else {
            throw TeacherKLBenchmarkError.logitsShapeMismatch(
                sampleID: quality.id,
                teacher: teacherLogits.shape,
                student: studentLogits.shape
            )
        }
        MLX.eval(studentLogits)

        let studentLossSum = MLXFast.crossEntropy(
            logits: studentLogits,
            targets: targets
        ).sum()
        let teacherPrediction = teacherLogits.argMax(axis: -1)
        let studentPrediction = studentLogits.argMax(axis: -1)
        let agreement = (teacherPrediction .== studentPrediction).asType(.int32).sum()
        let teacherCorrectMask = teacherPrediction .== targets
        let studentCorrectMask = studentPrediction .== targets
        let teacherCorrect = teacherCorrectMask.asType(.int32).sum()
        let studentCorrect = studentCorrectMask.asType(.int32).sum()
        let regressions = (teacherCorrectMask .&& logicalNot(studentCorrectMask))
            .asType(.int32).sum()
        let improvements = (logicalNot(teacherCorrectMask) .&& studentCorrectMask)
            .asType(.int32).sum()
        MLX.eval(
            studentLossSum,
            agreement,
            teacherCorrect,
            studentCorrect,
            regressions,
            improvements
        )
        guard Int(teacherCorrect.item(Int32.self)) == teacherSample.groundTruthTop1CorrectCount
        else {
            throw TeacherKLBenchmarkError.invalidMeasurement(
                sampleID: quality.id,
                detail: "cached teacher top-1 count does not match cached teacher logits"
            )
        }

        var sampleKLSum = 0.0
        var start = 0
        while start < inputCount {
            let end = min(inputCount, start + positionChunkSize)
            let teacherChunk = teacherLogits[0..., start..<end, 0...].asType(.float32)
            let studentChunk = studentLogits[0..., start..<end, 0...].asType(.float32)
            let teacherLogProbability = logSoftmax(teacherChunk, axis: -1)
            let studentLogProbability = logSoftmax(studentChunk, axis: -1)
            let teacherProbability = exp(teacherLogProbability)
            let chunkKL = (teacherProbability * (teacherLogProbability - studentLogProbability))
                .sum(axis: -1).sum()
            MLX.eval(chunkKL)
            sampleKLSum += Double(chunkKL.item(Float.self))
            start = end
        }
        guard sampleKLSum.isFinite, sampleKLSum >= -1e-5 * Double(inputCount) else {
            throw TeacherKLBenchmarkError.invalidMeasurement(
                sampleID: quality.id,
                detail: "KL sum is non-finite or materially negative: \(sampleKLSum)"
            )
        }
        sampleKLSum = max(0, sampleKLSum)

        let studentQuality = try ModelQualitySampleResult(
            id: quality.id,
            category: quality.category,
            originalTokenCount: quality.originalTokenCount,
            evaluatedTokenCount: quality.evaluatedTokenCount,
            tokenIDFingerprint: quality.tokenIDFingerprint,
            nllSum: Double(studentLossSum.item(Float.self))
        )
        let agreementCount = Int(agreement.item(Int32.self))
        let studentCorrectCount = Int(studentCorrect.item(Int32.self))
        let regressionCount = Int(regressions.item(Int32.self))
        let improvementCount = Int(improvements.item(Int32.self))
        let paired = PairedSampleResult(
            id: quality.id,
            category: quality.category,
            scoredTokenCount: quality.scoredTokenCount,
            tokenIDFingerprint: quality.tokenIDFingerprint,
            teacherNLL: quality.nll,
            studentNLL: studentQuality.nll,
            studentMinusTeacherNLL: studentQuality.nll - quality.nll,
            teacherKLSum: sampleKLSum,
            teacherKL: sampleKLSum / Double(quality.scoredTokenCount),
            teacherStudentTop1AgreementCount: agreementCount,
            teacherStudentTop1Agreement:
                Double(agreementCount) / Double(quality.scoredTokenCount),
            teacherGroundTruthTop1CorrectCount: teacherSample.groundTruthTop1CorrectCount,
            studentGroundTruthTop1CorrectCount: studentCorrectCount,
            teacherCorrectStudentWrongCount: regressionCount,
            teacherWrongStudentCorrectCount: improvementCount
        )
        return (studentQuality, paired)
    }

    private func validateInputs(
        teacherURL: URL,
        studentURLs: [URL],
        corpusURL: URL,
        outputURL: URL
    ) throws {
        try requireDirectory(teacherURL, label: "teacher")
        guard checkpointDeclaredDType(teacherURL)?.lowercased() == "bfloat16" else {
            throw TeacherKLBenchmarkError.invalidInput(
                "teacher config.json must declare dtype=bfloat16: \(teacherURL.path)"
            )
        }
        guard !checkpointIsQuantized(teacherURL) || allowQuantizedReference else {
            throw TeacherKLBenchmarkError.invalidInput(
                "teacher must be unquantized BF16 unless --allow-quantized-reference explicitly accepts provisional fidelity: \(teacherURL.path)"
            )
        }
        for studentURL in studentURLs {
            try requireDirectory(studentURL, label: "student")
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: corpusURL.path, isDirectory: &isDirectory),
            !isDirectory.boolValue
        else {
            throw TeacherKLBenchmarkError.invalidInput(
                "corpus file does not exist: \(corpusURL.path)"
            )
        }
        guard corpusURL != outputURL else {
            throw TeacherKLBenchmarkError.invalidInput("output must not replace the corpus")
        }
        if FileManager.default.fileExists(atPath: outputURL.path), !overwrite {
            throw TeacherKLBenchmarkError.invalidInput(
                "output already exists (pass --overwrite to replace it): \(outputURL.path)"
            )
        }
        if let teacherCacheDirectory {
            let cacheURL = localURL(teacherCacheDirectory, isDirectory: true)
            guard !FileManager.default.fileExists(atPath: cacheURL.path) else {
                throw TeacherKLBenchmarkError.invalidInput(
                    "teacher cache directory already exists: \(cacheURL.path)"
                )
            }
            guard !isDescendant(cacheURL, of: teacherURL),
                studentURLs.allSatisfy({ !isDescendant(cacheURL, of: $0) })
            else {
                throw TeacherKLBenchmarkError.invalidInput(
                    "teacher cache directory must not be inside a checkpoint"
                )
            }
            guard !isDescendant(outputURL, of: cacheURL) else {
                throw TeacherKLBenchmarkError.invalidInput(
                    "output report must not be inside the temporary teacher cache"
                )
            }
        }
    }

    private func createTeacherCacheDirectory() throws -> URL {
        let url: URL
        if let teacherCacheDirectory {
            url = localURL(teacherCacheDirectory, isDirectory: true)
        } else {
            url = FileManager.default.temporaryDirectory.appendingPathComponent(
                "model-runner-teacher-kl-\(UUID().uuidString)",
                isDirectory: true
            )
        }
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }
}

private func requireDirectory(_ url: URL, label: String) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
        isDirectory.boolValue
    else {
        throw TeacherKLBenchmarkError.invalidInput(
            "\(label) directory does not exist: \(url.path)"
        )
    }
}

private func localURL(_ path: String, isDirectory: Bool = false) -> URL {
    let expanded = NSString(string: path).expandingTildeInPath
    return URL(fileURLWithPath: expanded, isDirectory: isDirectory).standardizedFileURL
}

private func checkpointModelType(_ modelURL: URL) -> String? {
    checkpointConfig(modelURL)?["model_type"] as? String
        ?? (checkpointConfig(modelURL)?["text_config"] as? [String: Any])?["model_type"] as? String
}

private func checkpointDeclaredDType(_ modelURL: URL) -> String? {
    let config = checkpointConfig(modelURL)
    return config?["dtype"] as? String
        ?? config?["torch_dtype"] as? String
        ?? (config?["text_config"] as? [String: Any])?["dtype"] as? String
}

private func checkpointIsQuantized(_ modelURL: URL) -> Bool {
    guard let config = checkpointConfig(modelURL) else {
        return false
    }
    if hasNonNullValue(config, key: "quantization")
        || hasNonNullValue(config, key: "quantization_config")
    {
        return true
    }
    guard let textConfig = config["text_config"] as? [String: Any] else {
        return false
    }
    return hasNonNullValue(textConfig, key: "quantization")
        || hasNonNullValue(textConfig, key: "quantization_config")
}

private func hasNonNullValue(_ object: [String: Any], key: String) -> Bool {
    guard let value = object[key] else {
        return false
    }
    return !(value is NSNull)
}

private func checkpointConfig(_ modelURL: URL) -> [String: Any]? {
    let configURL = modelURL.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
        let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
        return nil
    }
    return object
}

private func isDescendant(_ candidate: URL, of directory: URL) -> Bool {
    let directoryPath = directory.standardizedFileURL.path
    let prefix = directoryPath.hasSuffix("/") ? directoryPath : directoryPath + "/"
    return candidate.standardizedFileURL.path.hasPrefix(prefix)
}

private let checkpointFingerprintMethod = "fnv1a64-path-size-and-sampled-content-v1"

private func checkpointFingerprint(_ directory: URL) throws -> String {
    let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey]
    guard
        let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        )
    else {
        throw TeacherKLBenchmarkError.invalidInput(
            "cannot enumerate checkpoint: \(directory.path)"
        )
    }

    var files = [URL]()
    for case let fileURL as URL in enumerator {
        let values = try fileURL.resourceValues(forKeys: Set(keys))
        if values.isRegularFile == true {
            files.append(fileURL)
        }
    }
    files.sort { relativePath($0, under: directory) < relativePath($1, under: directory) }

    var hasher = FNV1a64()
    hasher.update(UInt64(files.count))
    let sampleBytes = 64 * 1_024
    for fileURL in files {
        let relative = relativePath(fileURL, under: directory)
        let size = try fileByteCount(fileURL)
        hasher.update(relative)
        hasher.update(UInt64(size))
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }
        if size <= sampleBytes * 3 {
            hasher.update(try handle.readToEnd() ?? Data())
        } else {
            let offsets = [0, max(0, size / 2 - sampleBytes / 2), size - sampleBytes]
            for offset in offsets {
                try handle.seek(toOffset: UInt64(offset))
                hasher.update(try handle.read(upToCount: sampleBytes) ?? Data())
            }
        }
    }
    return hasher.fingerprint
}

private func teacherCacheFingerprint(_ manifest: TeacherCacheManifest) -> String {
    var hasher = FNV1a64()
    hasher.update(UInt64(manifest.format))
    hasher.update(manifest.strategy)
    hasher.update(manifest.logitsDType)
    hasher.update(manifest.teacherCheckpointFingerprint)
    hasher.update(manifest.corpusFingerprint)
    hasher.update(manifest.tokenIDFingerprint)
    for sample in manifest.samples {
        hasher.update(sample.quality.id)
        hasher.update(sample.quality.tokenIDFingerprint)
        hasher.update(sample.logitsFile)
        hasher.update(UInt64(sample.logitsBytes))
        for dimension in sample.logitsShape {
            hasher.update(UInt64(dimension))
        }
    }
    return hasher.fingerprint
}

private func relativePath(_ url: URL, under directory: URL) -> String {
    let prefix = directory.path.hasSuffix("/") ? directory.path : directory.path + "/"
    guard url.path.hasPrefix(prefix) else {
        return url.lastPathComponent
    }
    return String(url.path.dropFirst(prefix.count))
}

private func fileByteCount(_ url: URL) throws -> Int {
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    guard let size = attributes[.size] as? NSNumber else {
        throw TeacherKLBenchmarkError.invalidInput("cannot read file size: \(url.path)")
    }
    return size.intValue
}

private func directoryByteCount(_ url: URL) throws -> Int {
    guard
        let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]
        )
    else {
        return 0
    }
    var total = 0
    for case let fileURL as URL in enumerator {
        let values = try fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        if values.isRegularFile == true {
            let (next, overflow) = total.addingReportingOverflow(values.fileSize ?? 0)
            guard !overflow else {
                throw TeacherKLBenchmarkError.invalidInput("teacher cache byte count overflow")
            }
            total = next
        }
    }
    return total
}

private func writeJSON(_ value: some Encodable, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(value).write(to: url, options: .atomic)
}

private func seconds(_ duration: Duration) -> Double {
    let components = duration.components
    return Double(components.seconds)
        + Double(components.attoseconds) / 1_000_000_000_000_000_000
}

private func mib(_ bytes: Int) -> Double {
    Double(bytes) / 1_048_576
}

private struct FNV1a64 {
    private var value: UInt64 = 0xcbf2_9ce4_8422_2325

    mutating func update(byte: UInt8) {
        value ^= UInt64(byte)
        value &*= 0x100_0000_01b3
    }

    mutating func update(_ integer: UInt64) {
        for shift in stride(from: 0, through: 56, by: 8) {
            update(byte: UInt8(truncatingIfNeeded: integer >> UInt64(shift)))
        }
    }

    mutating func update(_ string: String) {
        let bytes = Array(string.utf8)
        update(UInt64(bytes.count))
        for byte in bytes {
            update(byte: byte)
        }
    }

    mutating func update(_ data: Data) {
        update(UInt64(data.count))
        for byte in data {
            update(byte: byte)
        }
    }

    var fingerprint: String {
        let hex = String(value, radix: 16)
        return "fnv1a64:" + String(repeating: "0", count: 16 - hex.count) + hex
    }
}

private var backendName: String {
    #if MLX_METAL_BACKEND
        "metal"
    #elseif MLX_CUDA_BACKEND
        "cuda"
    #elseif MLX_CPU_BACKEND
        "cpu"
    #else
        "unknown"
    #endif
}

private var benchmarkEngine: ModelEngine {
    #if MLX_METAL_BACKEND
        .metal
    #elseif MLX_CUDA_BACKEND
        .cuda
    #else
        .cpu
    #endif
}
