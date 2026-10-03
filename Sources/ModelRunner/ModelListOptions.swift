import ArgumentParser

/// ArgumentParser consumes parent options before descending into subcommands.
/// A shared option group carries the parsed --list value to the selected child;
/// independent flags with the same spelling silently lose the child's value.
/// Keep the documented alias here too, so it can never be swallowed before
/// a destructive subcommand decides whether the user requested listing.
struct ModelListOptions: ParsableArguments {
    @Flag(name: [.long, .customLong("list-models")], help: "List models for the selected command and exit")
    var list = false
}
