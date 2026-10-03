import ArgumentParser

struct ListenerOptions: ParsableArguments {
    @Option(name: .long, help: "Address to listen on")
    var host: String?

    @Option(name: .shortAndLong, help: "Port to listen on")
    var port: Int?

    @Option(name: .long, help: "Execution engine: auto, metal, cuda, or cpu")
    var engine: String?

    @Option(name: .shortAndLong, help: "Path to model-stack settings JSON")
    var config: String?

    @Flag(name: .long, help: "Log incoming requests, generation settings, and request outcomes")
    var verbose = false

    @Flag(name: .long, help: "Use plain server logs instead of the interactive terminal console")
    var noUI = false
}

struct ModelSelectionOptions: ParsableArguments {
    @Option(name: .shortAndLong, help: "Model name in ~/.midnight/models or an MLX folder")
    var model: String?

    @Option(name: .long, help: "Model name exposed by the endpoint")
    var name: String?

    @Option(name: .long, help: "Path to an MLX LoRA adapter folder")
    var adapter: String?

    @Option(name: .long, help: "Override the LoRA adapter scale")
    var adapterScale: Float?

    @Flag(name: .long, help: "Start the server with no model loaded")
    var idle = false
}

struct GenerationOptions: ParsableArguments {
    @Option(name: .long, help: "Default and hard maximum generated tokens per request")
    var maxTokens: Int?

    @Option(name: .long, help: "Maximum prompt plus output tokens (cannot exceed model context)")
    var contextLength: Int?

    @Option(name: .long, help: "Maximum tokens per prefill chunk (1...8192; default 512)")
    var prefillStepSize: Int?

    @Option(name: .long, help: "Experimental KV compression: none, affine8, affine4, turbo8v4")
    var kvCompression: String?
}

struct SpeculativeDecodingOptions: ParsableArguments {
    @Flag(name: .long, help: "Disable automatic discovery of installed speculative assistants")
    var noAutoAssistant = false

    @Option(
        name: .long,
        help:
            "Path to a compatible DFlash drafter checkpoint (experimental; Laguna supports target-verified sampling; Muse requires temperature 0; output may differ from target-only decoding)"
    )
    var dflashModel: String?

    @Option(name: .long, help: "DFlash verification block size (2...checkpoint maximum; default Laguna 3, Muse 4)")
    var dflashBlockSize: Int?

    @Option(
        name: .long,
        help:
            "Gemma 4 assistant checkpoint (experimental; supports target-verified sampling; output may differ from target-only decoding)"
    )
    var gemmaAssistantModel: String?

    @Option(name: .long, help: "Gemma assistant verification block size (2...4)")
    var gemmaAssistantBlockSize: Int?

    @Option(name: .long, help: "Quantize an unquantized Gemma assistant in memory (4 or 8 bits)")
    var gemmaAssistantQuantizationBits: Int?
}
