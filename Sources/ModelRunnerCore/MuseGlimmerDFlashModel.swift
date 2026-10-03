import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import ModelRunnerProtocol

/// Native implementation of the published Muse Glimmer block-diffusion
/// assistant.  It deliberately has no embedding or LM head: both operations
/// remain owned by the target, which makes target verification lossless.
struct MuseGlimmerAssistantConfiguration: Decodable, Sendable {
    let modelType: String
    let architectures: [String]
    let blockSize: Int
    let maskTokenID: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let keyValueHeads: Int
    let headDimension: Int
    let rmsNormEpsilon: Float
    let ropeTheta: Float
    let slidingWindow: Int
    let layerTypes: [String]
    let targetLayerIDs: [Int]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case blockSize = "block_size"
        case maskTokenID = "mask_token_id"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDimension = "head_dim"
        case rmsNormEpsilon = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case targetLayerIDs = "target_layer_ids"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        architectures = try c.decode([String].self, forKey: .architectures)
        blockSize = try c.decode(Int.self, forKey: .blockSize)
        maskTokenID = try c.decode(Int.self, forKey: .maskTokenID)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        keyValueHeads = try c.decode(Int.self, forKey: .keyValueHeads)
        headDimension = try c.decode(Int.self, forKey: .headDimension)
        rmsNormEpsilon = try c.decode(Float.self, forKey: .rmsNormEpsilon)
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        targetLayerIDs = try c.decode([Int].self, forKey: .targetLayerIDs)
        let rope = try c.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .ropeParameters)
        ropeTheta = try rope.decode(Float.self, forKey: DynamicCodingKey("rope_theta"))
        guard modelType == "muse_glimmer_assistant",
            architectures.contains("MuseGlimmerAssistantModel"), blockSize >= 2,
            hiddenSize > 0, hiddenLayers > 0, attentionHeads.isMultiple(of: keyValueHeads),
            layerTypes.count == hiddenLayers, layerTypes.allSatisfy({ $0 == "sliding_attention" })
        else {
            throw RequestAdmissionError.configuration("Invalid Muse Glimmer assistant configuration")
        }
    }
}

private struct DynamicCodingKey: CodingKey {
    var stringValue: String
    init(_ value: String) { stringValue = value }
    init?(stringValue: String) { self.stringValue = stringValue }
    var intValue: Int? { nil }
    init?(intValue: Int) { nil }
}

final class MuseDFlashAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var o: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: MuseGlimmerAssistantConfiguration) {
        heads = c.attentionHeads
        kvHeads = c.keyValueHeads
        headDim = c.headDimension
        scale = pow(Float(c.headDimension), -0.5)
        rope = initializeRope(
            dims: c.headDimension, base: c.ropeTheta, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: 131_072)
        _q.wrappedValue = Linear(c.hiddenSize, c.attentionHeads * c.headDimension, bias: false)
        _k.wrappedValue = Linear(c.hiddenSize, c.keyValueHeads * c.headDimension, bias: false)
        _v.wrappedValue = Linear(c.hiddenSize, c.keyValueHeads * c.headDimension, bias: false)
        _o.wrappedValue = Linear(c.attentionHeads * c.headDimension, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: c.headDimension, eps: c.rmsNormEpsilon)
        _kNorm.wrappedValue = RMSNorm(dimensions: c.headDimension, eps: c.rmsNormEpsilon)
        super.init()
    }

    func appendContext(_ context: MLXArray, cache: KVCache, position: Int) {
        let b = context.dim(0)
        let n = context.dim(1)
        var keys = kNorm(k(context).reshaped(b, n, kvHeads, headDim)).transposed(0, 2, 1, 3)
        keys = applyRotaryPosition(rope, to: keys, offset: .scalar(position))
        let values = v(context).reshaped(b, n, kvHeads, headDim).transposed(0, 2, 1, 3)
        _ = cache.update(keys: keys, values: values)
    }

    func callAsFunction(_ hidden: MLXArray, cache: KVCache, position: Int) -> MLXArray {
        let b = hidden.dim(0)
        let n = hidden.dim(1)
        var queries = qNorm(q(hidden).reshaped(b, n, heads, headDim)).transposed(0, 2, 1, 3)
        var keys = kNorm(k(hidden).reshaped(b, n, kvHeads, headDim)).transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, offset: .scalar(position))
        keys = applyRotaryPosition(rope, to: keys, offset: .scalar(position))
        let values = v(hidden).reshaped(b, n, kvHeads, headDim).transposed(0, 2, 1, 3)
        let state = cache.state
        let allKeys = state.count == 2 ? concatenated([state[0], keys], axis: 2) : keys
        let allValues = state.count == 2 ? concatenated([state[1], values], axis: 2) : values
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: allKeys,
            values: allValues, scale: scale, mask: .none)
        return o(output.transposed(0, 2, 1, 3).reshaped(b, n, heads * headDim))
    }
}

final class MuseDFlashLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MuseDFlashAttention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: MuseDFlashMLP
    init(_ c: MuseGlimmerAssistantConfiguration) {
        _attention.wrappedValue = MuseDFlashAttention(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEpsilon)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEpsilon)
        _mlp.wrappedValue = MuseDFlashMLP(c)
        super.init()
    }
    func appendContext(_ x: MLXArray, cache: KVCache, position: Int) {
        attention.appendContext(x, cache: cache, position: position)
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache, position: Int) -> MLXArray {
        let h = x + attention(inputNorm(x), cache: cache, position: position)
        return h + mlp(postNorm(h))
    }
}

final class MuseDFlashMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: MuseGlimmerAssistantConfiguration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class MuseGlimmerDFlashModel: Module, StatefulMTPDrafterModel {
    let configuration: MuseGlimmerAssistantConfiguration
    let maximumBlockSize: Int?
    let requiresSharedTargetKV = false
    let requiresPromptPrefill = true
    let requiresGreedySampling = true
    var promptHiddenStateWindow: Int? { configuration.slidingWindow }
    let supportsChunkedPromptPrefill = true
    @ModuleInfo(key: "encoder") var encoder: MuseDFlashEncoder
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [MuseDFlashLayer]
    init(_ c: MuseGlimmerAssistantConfiguration) {
        configuration = c
        maximumBlockSize = c.blockSize
        _encoder.wrappedValue = MuseDFlashEncoder(c)
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEpsilon)
        _layers.wrappedValue = (0..<c.hiddenLayers).map { _ in MuseDFlashLayer(c) }
        super.init()
    }
    var targetLayerIDs: [Int] { configuration.targetLayerIDs }
    var captureWindow: Int { configuration.slidingWindow }
    func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: layers.map { _ in RotatingKVCache(maxSize: configuration.slidingWindow) })
    }
    func prepareDrafterState(
        target _: any LanguageModel, promptTokens: MLXArray, targetHidden: MLXArray,
        firstBonus _: MLXArray, positionDeltas _: MLXArray?, state: inout MTPDrafterState, sampler _: any LogitSampler
    ) {
        state.nextPosition = promptTokens.dim(-1) - targetHidden.dim(1)
        append(targetHidden, state: &state)
    }
    func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)], positionDeltas _: MLXArray?, queryOffset: Int,
        blockSize: Int, state: inout MTPDrafterState, sampler: any LogitSampler
    ) -> MLXArray {
        guard let target = target as? MuseGlimmer else {
            fatalError("Muse DFlash requires MuseGlimmer target")
        }
        precondition(queryOffset == state.nextPosition && blockSize >= 2 && blockSize <= configuration.blockSize)
        let anchor = lastToken.ndim == 1 ? lastToken.reshaped(lastToken.dim(0), 1) : lastToken
        let masks = MLXArray.full(
            [anchor.dim(0), blockSize - 1], values: MLXArray(Int32(configuration.maskTokenID)), dtype: .int32)
        var h = target.dflashTokenEmbeddings(concatenated([anchor.asType(.int32), masks], axis: 1))
        for (layer, cache) in zip(layers, state.cache) {
            h = layer(h, cache: cache, position: queryOffset)
        }
        return sampler.sample(logits: target.dflashRawLogits(norm(h[0..., 1..., 0...])))
    }
    func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?, queryOffset: Int,
        blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        state.nextPosition = queryOffset - lastHidden.dim(1)
        append(lastHidden, state: &state)
        return draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize, state: &state,
            sampler: sampler)
    }
    func commitDrafterState(
        target _: any LanguageModel, targetHidden: MLXArray, draftTokens _: MLXArray,
        acceptedCount: Int, finalToken _: MLXArray, positionDeltas _: MLXArray?, state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) { append(targetHidden[0..., ..<(acceptedCount + 1), 0...], state: &state) }
    private func append(_ targetHidden: MLXArray, state: inout MTPDrafterState) {
        guard targetHidden.dim(1) > 0 else {
            return
        }
        precondition(targetHidden.dim(-1) == configuration.hiddenSize * configuration.targetLayerIDs.count)
        let context = encoder(targetHidden)
        for (layer, cache) in zip(layers, state.cache) {
            layer.appendContext(context, cache: cache, position: state.nextPosition)
        }
        state.nextPosition += targetHidden.dim(1)
    }
}

final class MuseDFlashEncoder: Module {
    @ModuleInfo(key: "fc") var projection: Linear
    @ModuleInfo(key: "output_norm_enc") var norm: RMSNorm
    init(_ c: MuseGlimmerAssistantConfiguration) {
        _projection.wrappedValue = Linear(c.hiddenSize * c.targetLayerIDs.count, c.hiddenSize, bias: false)
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEpsilon)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(projection(x)) }
}

/// Registers the Muse Glimmer DFlash drafter once for runtime loads.
public enum MuseGlimmerDFlashRegistration {
    private actor Registrar {
        var registered = false
        func register() async {
            guard !registered else {
                return
            }
            registered = true
            await MTPDrafterTypeRegistry.shared.registerModelType(
                "muse_glimmer_assistant",
                matches: { data in
                    guard let o = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                        return false
                    }
                    return o["model_type"] as? String == "muse_glimmer_assistant"
                },
                creator: { data in
                    MuseGlimmerDFlashModel(try JSONDecoder().decode(MuseGlimmerAssistantConfiguration.self, from: data))
                })
        }
    }
    private static let registrar = Registrar()
    /// Registers the drafter once with the process-wide model type registry.
    public static func register() async { await registrar.register() }
}
