import Foundation
import ModelRunnerProtocol
import Testing

@Suite("OpenAI and Mistral speech wire formats")
struct AudioSpeechWireTests {
    @Test("Decodes the official OpenAI speech shape and defaults")
    func openAIRequest() throws {
        let request = try AudioSpeechRequest.decode(
            from: Data(#"{"model":"voxtral","input":"Hello","voice":{"id":"casual_female"}}"#.utf8)
        )
        guard case .openAI(let speech) = request else {
            Issue.record("Expected an OpenAI request")
            return
        }
        #expect(speech.model == "voxtral")
        #expect(speech.voice == .id("casual_female"))
        #expect(speech.responseFormat == .mp3)
        #expect(speech.streamFormat == .audio)
        #expect(speech.speed == 1)
    }

    @Test("Decodes every official Mistral speech field")
    func mistralRequest() throws {
        let data = Data(
            #"{"input":"Bonjour","model":"voxtral","voice_id":"fr_female","ref_audio":"AQID","response_format":"pcm","stream":true,"metadata":{"turn":3},"prompt_cache_key":"key"}"#.utf8
        )
        let request = try AudioSpeechRequest.decode(from: data)
        guard case .mistral(let speech) = request else {
            Issue.record("Expected a Mistral request")
            return
        }
        #expect(speech.input == "Bonjour")
        #expect(speech.model == "voxtral")
        #expect(speech.voiceID == "fr_female")
        #expect(speech.refAudio == "AQID")
        #expect(speech.responseFormat == .pcm)
        #expect(speech.stream)
        #expect(speech.metadata?["turn"] == .integer(3))
        #expect(speech.promptCacheKey == "key")
    }

    @Test("A minimal request is Mistral and mixed dialect fields fail")
    func discrimination() throws {
        let minimal = try AudioSpeechRequest.decode(
            from: Data(#"{"input":"Hello","model":"voxtral"}"#.utf8)
        )
        guard case .mistral = minimal else {
            Issue.record("A request without OpenAI's required voice belongs to Mistral")
            return
        }
        let mistralWithAdditionalProperty = try AudioSpeechRequest.decode(
            from: Data(#"{"input":"Hello","instructions":"additional Mistral property"}"#.utf8)
        )
        guard case .mistral = mistralWithAdditionalProperty else {
            Issue.record("Mistral permits additional request properties")
            return
        }

        #expect(throws: AudioSpeechRequestError.mixedProtocols) {
            try AudioSpeechRequest.decode(
                from: Data(#"{"model":"voxtral","input":"Hello","voice":"alloy","voice_id":"casual_female"}"#.utf8)
            )
        }
    }

    @Test("Supports all documented output formats")
    func formats() throws {
        for value in ["pcm", "wav", "mp3", "flac", "opus"] {
            let request = try JSONDecoder().decode(
                MistralSpeechRequest.self,
                from: Data("{\"input\":\"x\",\"response_format\":\"\(value)\"}".utf8)
            )
            #expect(request.responseFormat?.rawValue == value)
        }
        for value in ["mp3", "opus", "aac", "flac", "wav", "pcm"] {
            let request = try JSONDecoder().decode(
                OpenAISpeechRequest.self,
                from: Data(
                    "{\"model\":\"x\",\"input\":\"x\",\"voice\":\"alloy\",\"response_format\":\"\(value)\"}".utf8
                )
            )
            #expect(request.responseFormat.rawValue == value)
        }
    }

    @Test("Rejects explicit null for defaulted non-null fields")
    func nonNullDefaults() {
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(
                MistralSpeechRequest.self,
                from: Data(#"{"input":"x","stream":null}"#.utf8)
            )
        }
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(
                MistralSpeechRequest.self,
                from: Data(#"{"input":"x","response_format":null}"#.utf8)
            )
        }
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(
                OpenAISpeechRequest.self,
                from: Data(#"{"model":"x","input":"x","voice":"alloy","speed":null}"#.utf8)
            )
        }
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(
                VoiceCreateRequest.self,
                from: Data(#"{"name":"Demo","sample_audio":"AQID","languages":null}"#.utf8)
            )
        }
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(
                VoiceCreateRequest.self,
                from: Data(#"{"name":"Demo","sample_audio":"AQID","retention_notice":null}"#.utf8)
            )
        }
    }

    @Test("Encodes Mistral nonstream and named-event data exactly")
    func responses() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        #expect(
            String(decoding: try encoder.encode(MistralSpeechResponse(audioData: "AQID")), as: UTF8.self)
                == #"{"audio_data":"AQID"}"#
        )
        #expect(
            String(
                decoding: try encoder.encode(MistralSpeechAudioDeltaEvent(audioData: "AQID")),
                as: UTF8.self
            ) == #"{"audio_data":"AQID","type":"speech.audio.delta"}"#
        )
        let done = MistralSpeechAudioDoneEvent(
            usage: .init(promptTokens: 4, totalTokens: 14, completionTokens: 10)
        )
        let doneJSON = String(decoding: try encoder.encode(done), as: UTF8.self)
        #expect(doneJSON.contains(#""type":"speech.audio.done""#))
        #expect(doneJSON.contains(#""prompt_tokens":4"#))
        #expect(doneJSON.contains(#""completion_tokens":10"#))
        #expect(doneJSON.contains(#""total_tokens":14"#))
    }

    @Test("Encodes official OpenAI speech SSE event data exactly")
    func openAISSEEvents() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        let delta = OpenAISpeechAudioDeltaEvent(audio: "AQID")
        #expect(
            String(decoding: try encoder.encode(delta), as: UTF8.self)
                == #"{"audio":"AQID","type":"speech.audio.delta"}"#
        )

        let done = OpenAISpeechAudioDoneEvent(
            usage: .init(inputTokens: 4, outputTokens: 10, totalTokens: 14)
        )
        #expect(
            String(decoding: try encoder.encode(done), as: UTF8.self)
                == #"{"type":"speech.audio.done","usage":{"input_tokens":4,"output_tokens":10,"total_tokens":14}}"#
        )
    }

    @Test("Voice creation follows official defaults and snake case")
    func createVoice() throws {
        let request = try JSONDecoder().decode(
            VoiceCreateRequest.self,
            from: Data(#"{"name":"Demo","sample_audio":"AQID","sample_filename":"demo.wav"}"#.utf8)
        )
        #expect(request.languages == [])
        #expect(request.retentionNotice == 30)
        #expect(request.sampleFilename == "demo.wav")
        let encoded = String(decoding: try JSONEncoder().encode(request), as: UTF8.self)
        #expect(encoded.contains(#""sample_audio":"AQID""#))
        #expect(encoded.contains(#""retention_notice":30"#))
    }

    @Test("Voice patches preserve missing, null, and value")
    func voicePatch() throws {
        let patch = try JSONDecoder().decode(
            VoiceUpdateRequest.self,
            from: Data(#"{"name":null,"languages":["fr"],"age":42}"#.utf8)
        )
        #expect(patch.name == .null)
        #expect(patch.languages == .value(["fr"]))
        #expect(patch.age == .value(42))
        #expect(patch.gender == .missing)

        let object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(patch)) as? [String: Any]
        )
        #expect(object.keys.contains("name"))
        #expect(object["name"] is NSNull)
        #expect(object["gender"] == nil)
    }

    @Test("Voice responses and validation errors match the SDK schema")
    func voiceAndErrors() throws {
        let voice = VoiceResponse(
            name: "Casual Female",
            id: "564f5854-5241-4c54-8000-000000000001",
            createdAt: Date(timeIntervalSince1970: 0),
            userID: nil,
            slug: "casual_female",
            languages: ["en"],
            gender: "female"
        )
        let list = VoiceListResponse(items: [voice], total: 1, page: 1, pageSize: 10, totalPages: 1)
        let encoded = try JSONEncoder().encode(list)
        let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(object["page"] as? Int == 1)
        #expect(object["page_size"] as? Int == 10)
        #expect(object["total_pages"] as? Int == 1)
        let item = try #require((object["items"] as? [[String: Any]])?.first)
        #expect(item["created_at"] as? String == "1970-01-01T00:00:00Z")
        #expect(item.keys.contains("user_id"))

        let validation = HTTPValidationError(detail: [
            .init(
                loc: [.string("body"), .string("input")],
                msg: "Field required",
                type: "missing"
            )
        ])
        let validationObject = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(validation)) as? [String: Any]
        )
        #expect(validationObject["detail"] is [[String: Any]])

        let localError = MistralError(message: "Not available", code: "unsupported_model_feature")
        let errorObject = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(localError)) as? [String: Any]
        )
        #expect(errorObject["object"] as? String == "error")
        #expect(errorObject["message"] as? String == "Not available")
        #expect(errorObject["error"] == nil)
    }
}
