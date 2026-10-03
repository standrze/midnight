import Foundation

/// A byte-level JSON grammar used to constrain sampling before a token is emitted.
/// Schema objects use deterministic, lexicographically sorted property order.
/// Unsupported schema constraints fail at construction instead of being ignored.
public struct StructuredOutputGrammar: Sendable {
    /// Incremental parser state after a valid prefix of output bytes.
    public struct State: Hashable, Sendable {
        fileprivate var stacks: [[Int]]
    }

    /// A response schema or grammar exceeds the supported constraints.
    public struct SchemaError: Error, LocalizedError, Sendable {
        public let message: String
        /// User-facing explanation for this error.
        public var errorDescription: String? { message }
    }

    private let rules: [[[Int]]]
    private let terminals: [ByteSet]
    public let initialState: State

    /// Compiles a supported JSON response format into a byte-level grammar.
    public init(responseFormat: OpenAIResponseFormat) throws {
        var compiler = Compiler()
        let root: Int
        switch responseFormat {
        case .text:
            throw SchemaError(message: "Text response format does not require a JSON grammar.")
        case .jsonObject:
            root = compiler.anyObject
        case .jsonSchema(let format):
            var parser = SchemaParser(root: format.schema, strict: format.strict == true)
            let schema = try parser.parse(format.schema)
            if case .union = schema {
                throw SchemaError(message: "json_schema root cannot be anyOf, including through a reference.")
            }
            guard schema.onlyObjects else {
                throw SchemaError(message: "json_schema must describe an object at the root.")
            }
            root = try compiler.compile(schema)
        }
        let document = compiler.rule([[compiler.whitespace, root, compiler.whitespace]])
        rules = compiler.rules
        terminals = compiler.terminals
        initialState = Self.normalized([[document]], rules: rules)
        guard !initialState.stacks.isEmpty else {
            throw SchemaError(message: "Response format exceeds grammar complexity limits.")
        }
    }

    /// Returns nil immediately when the bytes cannot extend any valid JSON document.
    public func advancing(_ state: State, bytes: [UInt8]) -> State? {
        var next = state
        for byte in bytes {
            guard let advanced = advancing(next, byte: byte) else {
                return nil
            }
            next = advanced
        }
        return next
    }

    /// Single-byte variant for shared-prefix vocabulary trie traversal.
    public func advancing(_ state: State, byte: UInt8) -> State? {
        var candidates = [[Int]]()
        for stack in state.stacks {
            guard let terminal = stack.last, terminal < 0,
                terminals[-terminal - 1].contains(byte)
            else {
                continue
            }
            candidates.append(Array(stack.dropLast()))
        }
        guard !candidates.isEmpty else {
            return nil
        }
        let result = Self.normalized(candidates, rules: rules)
        return result.stacks.isEmpty ? nil : result
    }

    /// Returns whether the consumed bytes can end as a complete JSON document.
    public func isComplete(_ state: State) -> Bool {
        state.stacks.contains(where: \.isEmpty)
    }

    /// Checks a complete UTF-8 string against this grammar.
    public func validate(_ output: String) -> Bool {
        guard let state = advancing(initialState, bytes: Array(output.utf8)) else {
            return false
        }
        return isComplete(state)
    }

    private static func normalized(_ candidates: [[Int]], rules: [[[Int]]]) -> State {
        var pending = candidates
        var seen = Set<[Int]>()
        var result = Set<[Int]>()
        while let stack = pending.popLast() {
            // Bound JSON nesting and parser work independently of the model's token limit.
            guard stack.count <= 2048, seen.insert(stack).inserted else {
                continue
            }
            // Fail closed if a schema creates too many simultaneous parser configurations.
            guard seen.count <= 16_384, result.count <= 4096, pending.count <= 16_384 else {
                return State(stacks: [])
            }
            guard let next = stack.last, next >= 0 else {
                result.insert(stack)
                continue
            }
            let prefix = stack.dropLast()
            for alternative in rules[next] {
                pending.append(Array(prefix) + Array(alternative.reversed()))
            }
        }
        return State(stacks: result.sorted { $0.lexicographicallyPrecedes($1) })
    }
}

private struct ByteSet: Hashable, Sendable {
    var words: [UInt64] = [0, 0, 0, 0]

    init(_ ranges: [ClosedRange<UInt8>]) {
        for range in ranges {
            for byte in range {
                words[Int(byte) / 64] |= UInt64(1) << (Int(byte) % 64)
            }
        }
    }

    func contains(_ byte: UInt8) -> Bool {
        words[Int(byte) / 64] & (UInt64(1) << (Int(byte) % 64)) != 0
    }
}

private indirect enum JSONSchema {
    struct Property: Sendable {
        let name: String
        let schema: JSONSchema
        let required: Bool
    }

    case any, null, boolean, string, number, integer
    case object([Property])
    case array(JSONSchema)
    case union([JSONSchema])
    case enumeration([OpenAIJSONValue])

    var onlyObjects: Bool {
        switch self {
        case .object: return true
        case .union(let alternatives): return alternatives.allSatisfy(\.onlyObjects)
        case .enumeration(let values):
            return values.allSatisfy {
                if case .object = $0 {
                    true
                } else {
                    false
                }
            }
        default: return false
        }
    }

    func accepts(_ value: OpenAIJSONValue) -> Bool {
        switch (self, value) {
        case (.any, _), (.null, .null), (.boolean, .bool), (.string, .string),
            (.integer, .integer), (.number, .integer):
            return true
        case (.number, .number(let number)): return number.isFinite
        case (.integer, .number(let number)): return number.isFinite && number.rounded() == number
        case (.array(let item), .array(let values)): return values.allSatisfy(item.accepts)
        case (.object(let properties), .object(let values)):
            let names = Set(properties.map(\.name))
            return Set(values.keys).isSubset(of: names)
                && properties.allSatisfy { property in
                    if let value = values[property.name] {
                        return property.schema.accepts(value)
                    }
                    return !property.required
                }
        case (.union(let schemas), _): return schemas.contains { $0.accepts(value) }
        case (.enumeration(let choices), _): return choices.contains(value)
        default: return false
        }
    }
}

extension JSONSchema: Sendable {}

private struct SchemaParser {
    let root: OpenAIJSONValue
    let strict: Bool
    private var count = 0
    private let annotations: Set<String> = ["title", "description", "$defs"]
    private let supported: Set<String> = [
        "type", "properties", "required", "additionalProperties", "items", "enum", "const",
        "anyOf", "$defs", "$ref", "title", "description",
    ]

    init(root: OpenAIJSONValue, strict: Bool) {
        self.root = root
        self.strict = strict
    }

    mutating func parse(
        _ value: OpenAIJSONValue, path: String = "$", depth: Int = 0, references: Set<String> = []
    ) throws -> JSONSchema {
        count += 1
        guard count <= 512, depth <= 32 else {
            throw error(path, "schema exceeds complexity limits")
        }
        guard case .object(let object) = value else {
            throw error(path, "schema must be an object")
        }
        if let key = object.keys.sorted().first(where: { !supported.contains($0) }) {
            throw error(path, "unsupported keyword '\(key)'")
        }
        for key in ["title", "description"] {
            if let value = object[key], case .string = value {
            } else if object[key] != nil {
                throw error(path, "\(key) must be a string")
            }
        }
        if let definitions = object["$defs"], case .object = definitions {
        } else if object["$defs"] != nil {
            throw error(path, "$defs must be an object")
        }
        if let reference = object["$ref"] {
            guard case .string(let reference) = reference, reference.hasPrefix("#/$defs/") else {
                throw error(path, "only local #/$defs/ references are supported")
            }
            guard Set(object.keys).subtracting(annotations).subtracting(["$ref"]).isEmpty else {
                throw error(path, "$ref cannot be combined with other constraints")
            }
            guard !references.contains(reference) else {
                throw error(path, "recursive $ref is unsupported")
            }
            var target = root
            for component in reference.dropFirst(2).split(separator: "/", omittingEmptySubsequences: false) {
                let key = component.replacingOccurrences(of: "~1", with: "/")
                    .replacingOccurrences(of: "~0", with: "~")
                guard case .object(let object) = target, let child = object[key] else {
                    throw error(path, "unresolved reference '\(reference)'")
                }
                target = child
            }
            return try parse(target, path: reference, depth: depth + 1, references: references.union([reference]))
        }
        if let alternatives = object["anyOf"] {
            guard depth != 0 else {
                throw error(path, "root anyOf is unsupported; use an object schema")
            }
            guard Set(object.keys).subtracting(annotations).subtracting(["anyOf"]).isEmpty else {
                throw error(path, "anyOf cannot be combined with other constraints")
            }
            guard case .array(let alternatives) = alternatives, !alternatives.isEmpty, alternatives.count <= 32 else {
                throw error(path, "anyOf must contain between 1 and 32 schemas")
            }
            var schemas = [JSONSchema]()
            for (index, alternative) in alternatives.enumerated() {
                schemas.append(
                    try parse(alternative, path: "\(path).anyOf[\(index)]", depth: depth + 1, references: references))
            }
            return .union(schemas)
        }

        var types = [String]()
        if let type = object["type"] {
            switch type {
            case .string(let name): types = [name]
            case .array(let names):
                for name in names {
                    guard case .string(let name) = name else {
                        throw error(path, "type entries must be strings")
                    }
                    types.append(name)
                }
                guard !types.isEmpty, Set(types).count == types.count else {
                    throw error(path, "type array must be nonempty and contain unique entries")
                }
            default: throw error(path, "type must be a string or string array")
            }
        }
        for (keys, type) in [(["properties", "required", "additionalProperties"], "object"), (["items"], "array")] {
            guard types.contains(type) || keys.allSatisfy({ object[$0] == nil }) else {
                throw error(path, "\(keys.joined(separator: ", ")) requires type '\(type)'")
            }
        }
        var schemas = [JSONSchema]()
        for type in types {
            switch type {
            case "null": schemas.append(.null)
            case "boolean": schemas.append(.boolean)
            case "string": schemas.append(.string)
            case "number": schemas.append(.number)
            case "integer": schemas.append(.integer)
            case "array":
                guard let items = object["items"] else {
                    throw error(path, "array requires an items schema")
                }
                schemas.append(
                    .array(try parse(items, path: "\(path).items", depth: depth + 1, references: references)))
            case "object":
                guard object["additionalProperties"] == .bool(false) else {
                    throw error(path, "object schemas require additionalProperties: false")
                }
                let properties: [String: OpenAIJSONValue]
                if let provided = object["properties"] {
                    guard case .object(let provided) = provided else {
                        throw error(path, "properties must be an object")
                    }
                    properties = provided
                } else {
                    properties = [:]
                }
                guard properties.count <= 128 else {
                    throw error(path, "too many object properties (maximum 128)")
                }
                var required = Set<String>()
                if let provided = object["required"] {
                    guard case .array(let provided) = provided else {
                        throw error(path, "required must be an array")
                    }
                    for name in provided {
                        guard case .string(let name) = name, required.insert(name).inserted,
                            properties[name] != nil
                        else {
                            throw error(path, "required entries must uniquely name declared properties")
                        }
                    }
                }
                guard !strict || required.count == properties.count else {
                    throw error(
                        path, "strict schemas must require every object property; express optional values with null")
                }
                var parsed = [JSONSchema.Property]()
                for name in properties.keys.sorted() {
                    guard name.utf8.count <= 1024 else {
                        throw error(path, "property name exceeds 1024 bytes")
                    }
                    parsed.append(
                        .init(
                            name: name,
                            schema: try parse(
                                properties[name]!, path: "\(path).properties.\(name)", depth: depth + 1,
                                references: references),
                            required: required.contains(name)
                        ))
                }
                schemas.append(.object(parsed))
            default: throw error(path, "unsupported type '\(type)'")
            }
        }
        let base: JSONSchema = schemas.isEmpty ? .any : schemas.count == 1 ? schemas[0] : .union(schemas)
        if object["enum"] != nil && object["const"] != nil {
            throw error(path, "combine enum and const in a single enum instead")
        }
        if let enumeration = object["enum"] {
            guard case .array(let values) = enumeration, !values.isEmpty, values.count <= 256 else {
                throw error(path, "enum must contain between 1 and 256 values")
            }
            let accepted = values.filter(base.accepts)
            guard !accepted.isEmpty else {
                throw error(path, "enum contains no values allowed by its schema")
            }
            return .enumeration(accepted)
        }
        if let constant = object["const"] {
            guard base.accepts(constant) else {
                throw error(path, "const does not satisfy its schema")
            }
            return .enumeration([constant])
        }
        return base
    }

    private func error(_ path: String, _ message: String) -> StructuredOutputGrammar.SchemaError {
        .init(message: "Invalid response_format schema at \(path): \(message).")
    }
}

private struct Compiler {
    var rules = [[[Int]]]()
    var terminals = [ByteSet]()
    private var terminalIDs = [ByteSet: Int]()
    private var encodedEnumBytes = 0
    var whitespace = 0
    var string = 0
    var integer = 0
    var number = 0
    var anyValue = 0
    var anyObject = 0

    init() {
        whitespace = rule([])
        rules[whitespace] = [[], [terminal([9...10, 13...13, 32...32]), whitespace]]
        let digits = terminal([48...57])
        let digitTail = rule([])
        rules[digitTail] = [[], [digits, digitTail]]
        let natural = rule([literal("0"), [terminal([49...57]), digitTail]])
        let sign = rule([[], literal("-")])
        integer = rule([[sign, natural]])
        let fraction = rule([[], literal(".") + [digits, digitTail]])
        let exponentSign = rule([[], literal("+"), literal("-")])
        let exponent = rule([[], [terminal([69...69, 101...101]), exponentSign, digits, digitTail]])
        number = rule([[integer, fraction, exponent]])

        let hex = terminal([48...57, 65...70, 97...102])
        let nonsurrogate = rule([
            [terminal([48...57, 65...67, 69...70, 97...99, 101...102]), hex, hex, hex],
            [terminal([68...68, 100...100]), terminal([48...55]), hex, hex],
        ])
        let surrogatePair = rule([
            [terminal([68...68, 100...100]), terminal([56...57, 65...66, 97...98]), hex, hex]
                + literal("\\u")
                + [terminal([68...68, 100...100]), terminal([67...70, 99...102]), hex, hex]
        ])
        let escape = rule([
            [terminal([34...34, 47...47, 92...92, 98...98, 102...102, 110...110, 114...114, 116...116])],
            literal("u") + [nonsurrogate], literal("u") + [surrogatePair],
        ])
        let continuation = terminal([128...191])
        let character = rule([
            [terminal([32...33, 35...91, 93...127])],
            literal("\\") + [escape],
            [terminal([194...223]), continuation],
            [terminal([224...224]), terminal([160...191]), continuation],
            [terminal([225...236, 238...239]), continuation, continuation],
            [terminal([237...237]), terminal([128...159]), continuation],
            [terminal([240...240]), terminal([144...191]), continuation, continuation],
            [terminal([241...243]), continuation, continuation, continuation],
            [terminal([244...244]), terminal([128...143]), continuation, continuation],
        ])
        let characters = rule([])
        rules[characters] = [[], [character, characters]]
        string = rule([literal("\"") + [characters] + literal("\"")])

        anyValue = rule([])
        anyObject = rule([])
        let member = rule([[string, whitespace] + literal(":") + [whitespace, anyValue, whitespace]])
        let members = rule([])
        rules[members] = [[], literal(",") + [whitespace, member, members]]
        rules[anyObject] = [
            literal("{") + [whitespace] + literal("}"),
            literal("{") + [whitespace, member, members] + literal("}"),
        ]
        let anyArray = array(item: anyValue)
        rules[anyValue] = [
            [string], [number], [anyObject], [anyArray], literal("true"), literal("false"), literal("null"),
        ]
    }

    mutating func compile(_ schema: JSONSchema) throws -> Int {
        switch schema {
        case .any: return anyValue
        case .null: return rule([literal("null")])
        case .boolean: return rule([literal("true"), literal("false")])
        case .string: return string
        case .integer: return integer
        case .number: return number
        case .array(let item): return array(item: try compile(item))
        case .union(let alternatives):
            var rules = [[Int]]()
            for alternative in alternatives {
                rules.append([try compile(alternative)])
            }
            return rule(rules)
        case .enumeration(let choices):
            var alternatives = [[Int]]()
            for choice in choices {
                let encoded = try encode(choice)
                encodedEnumBytes += encoded.utf8.count
                guard encodedEnumBytes <= 65_536 else {
                    throw StructuredOutputGrammar.SchemaError(
                        message: "Response format enum exceeds 65536 encoded bytes.")
                }
                alternatives.append(literal(encoded))
            }
            return rule(alternatives)
        case .object(let properties):
            var emptyTail = rule([[]])
            var populatedTail = emptyTail
            for property in properties.reversed() {
                let value = try compile(property.schema)
                let entry =
                    literal(try encode(.string(property.name))) + [whitespace]
                    + literal(":") + [whitespace, value, whitespace, populatedTail]
                let populated = literal(",") + [whitespace] + entry
                let nextEmpty = rule(property.required ? [entry] : [entry, [emptyTail]])
                let nextPopulated = rule(property.required ? [populated] : [populated, [populatedTail]])
                emptyTail = nextEmpty
                populatedTail = nextPopulated
            }
            return rule([literal("{") + [whitespace, emptyTail] + literal("}")])
        }
    }

    mutating func array(item: Int) -> Int {
        let tail = rule([])
        rules[tail] = [[], literal(",") + [whitespace, item, whitespace, tail]]
        return rule([
            literal("[") + [whitespace] + literal("]"),
            literal("[") + [whitespace, item, whitespace, tail] + literal("]"),
        ])
    }

    mutating func rule(_ alternatives: [[Int]]) -> Int {
        let id = rules.count
        rules.append(alternatives)
        return id
    }

    mutating func terminal(_ ranges: [ClosedRange<UInt8>]) -> Int {
        let set = ByteSet(ranges)
        if let id = terminalIDs[set] {
            return id
        }
        let id = -terminals.count - 1
        terminals.append(set)
        terminalIDs[set] = id
        return id
    }

    mutating func literal(_ text: String) -> [Int] {
        let symbols = text.utf8.map { terminal([$0...$0]) }
        guard symbols.count > 64 else {
            return symbols
        }
        // Long enum strings should not consume one parser stack frame per byte.
        var tail = [Int]()
        var end = symbols.count
        while end > 0 {
            let start = max(0, end - 64)
            tail = [rule([Array(symbols[start..<end]) + tail])]
            end = start
        }
        return tail
    }

    private func encode(_ value: OpenAIJSONValue) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}
