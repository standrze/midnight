import MLXLMCommon
import ModelRunnerProtocol

extension LongContextOptions {
    func apply(to parameters: inout GenerateParameters) throws {
        parameters.prefill.stepSize = prefillStepSize
        let strategy: KVCacheConfiguration.Strategy
        switch compression {
        case .none: return
        case .affine8: strategy = .affine(try .init(bits: 8))
        case .affine4: strategy = .affine(try .init(bits: 4))
        case .turbo8v4:
            strategy = .turboQuant(try .init(keyPrecision: .affineEightBit, valuePrecision: .fourBit))
        }
        parameters.kvCache = .init(strategy: strategy, compatibility: .requireAtLeastOneLayer)
    }
}
