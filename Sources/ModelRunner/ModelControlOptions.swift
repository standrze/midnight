import ArgumentParser

struct ModelControlOptions: ParsableArguments {
    @Option(name: .long, help: "Local Midnight base URL")
    var endpoint = "http://127.0.0.1:8080"
}
