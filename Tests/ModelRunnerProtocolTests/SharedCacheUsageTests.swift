import Foundation
import Testing
import ModelRunnerProtocol

@Suite struct SharedCacheUsageTests {
  @Test func openAIUsageReportsBoundedCachedTokens() throws {
    let usage = ChatCompletionUsage(promptTokens: 200, completionTokens: 8, cachedTokens: 128)
    let data = try JSONEncoder().encode(usage)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect((json["prompt_tokens_details"] as? [String: Int])?["cached_tokens"] == 128)
    #expect(try JSONDecoder().decode(ChatCompletionUsage.self, from: data) == usage)
    #expect(ChatCompletionUsage(promptTokens: 2, completionTokens: 1, cachedTokens: 99).promptTokensDetails?.cachedTokens == 2)
  }
  @Test func acceptsUsageWithoutDetails() throws {
    let data = Data(#"{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}"#.utf8)
    #expect(try JSONDecoder().decode(ChatCompletionUsage.self, from: data).promptTokensDetails == nil)
  }
}
