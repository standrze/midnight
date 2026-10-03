import Foundation
import MLX
import MLXLMCommon
import ModelRunnerProtocol

/// Native reference decision scorer shared with Afterglow through contract fixtures.
public enum ModelDecisionScoring {
    public static func prepare(
        request: DecisionRequest, contract: DecisionModelContract,
        tokenizer: any MLXLMCommon.Tokenizer
    ) throws -> [(name: String, tokens: [Int], candidates: [Int])] {
        try contract.validate()
        try request.validate(maximumChoices: contract.candidateCodes.count)
        return try request.names.map { name in
            let prompt = try DecisionPrompt.text(request: request, field: name, contract: contract)
            let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
            guard !tokens.isEmpty, tokens.count <= contract.maxLength else {
                throw DecisionError.invalidRequest(
                    "\(name): prompt has \(tokens.count) tokens; limit \(contract.maxLength). Nothing was truncated.")
            }
            let count = request.schema[name]!.values.count
            let candidates = Array(contract.candidateTokenIDs.prefix(count))
            for index in 0..<count {
                let appended = tokenizer.encode(text: prompt + contract.candidateCodes[index], addSpecialTokens: false)
                guard appended == tokens + [candidates[index]] else {
                    throw DecisionError.invalidContract(
                        "Candidate code \(contract.candidateCodes[index]) is not the contracted single token at the answer boundary."
                    )
                }
            }
            return (name, tokens, candidates)
        }
    }

    public static func score(
        context: ModelContext, request: DecisionRequest,
        contract: DecisionModelContract
    ) throws -> DecisionResponse {
        let prepared = try prepare(request: request, contract: contract, tokenizer: context.tokenizer)
        var fields: [String: DecisionFieldResult] = [:]
        var count = 0
        for row in prepared {
            try Task.checkCancellation()
            let cache = try context.model.newCache(parameters: nil)
            let logits = context.model(MLXArray(row.tokens).reshaped(1, row.tokens.count), cache: cache)
            guard logits.ndim == 3, logits.dim(0) == 1, logits.dim(1) > 0,
                row.candidates.allSatisfy({ $0 < logits.dim(2) })
            else {
                throw DecisionError.invalidContract("Model vocabulary or logits shape differs from decision contract.")
            }
            let candidates = take(logits[0, -1, 0...].asType(.float32), MLXArray(row.candidates), axis: 0)
            eval(candidates)
            let values = candidates.asArray(Float.self).map(Double.init)
            fields[row.name] = try DecisionScoring.result(
                logits: values,
                choices: request.schema[row.name]!.values, temperature: contract.temperature,
                rubric: (request.scoreFields ?? []).contains(row.name))
            count += row.tokens.count
        }
        try Task.checkCancellation()
        return DecisionResponse(
            model: request.model, revision: contract.sourceRevision ?? contract.revision,
            temperature: contract.temperature, fields: fields, promptTokens: count,
            artifactSHA256: contract.adapterFingerprint)
    }
}
