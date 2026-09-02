import MLX
import MLXNN

final class VoxtralAudioCodeEmbeddings: Module {
  @ModuleInfo(key: "embeddings") var embeddings: Embedding

  // This is generated index state, not a learned checkpoint parameter.
  private let runtimeConstants: VoxtralAudioEmbeddingRuntimeConstants

  init(configuration: VoxtralTTSConfiguration) {
    self._embeddings.wrappedValue = Embedding(
      embeddingCount: configuration.audioEmbeddingCount,
      dimensions: configuration.dimension
    )

    var values = [Int32(0)]
    let semanticSize = configuration.semanticCodebookSize + 2
    let acousticSize = configuration.acousticCodebookSize + 2
    for index in 0 ..< configuration.acousticCodebookCount {
      values.append(Int32(semanticSize + index * acousticSize))
    }
    self.runtimeConstants = VoxtralAudioEmbeddingRuntimeConstants(offsets: values)
    super.init()
  }

  func feedbackEmbedding(codes: MLXArray) -> MLXArray {
    embeddings(codes + runtimeConstants.offsets).sum(axis: 1, keepDims: true)
  }
}

private final class VoxtralAudioEmbeddingRuntimeConstants {
  let offsets: MLXArray

  init(offsets: [Int32]) {
    self.offsets = MLXArray(offsets).reshaped(1, offsets.count)
  }
}
