@_spi(Testing) import MLXLMCommon
import Testing

@Suite("Adaptive MTP draft policy")
struct MTPAdaptiveDraftPolicyTests {
    @Test("Useful tokens per second beats acceptance rate alone")
    func throughputChoice() {
        var policy = MTPAdaptiveDraftPolicy(maximumBlockSize: 8)
        // Width two yields 2/1 = 2 t/s; width four yields 4/4 = 1 t/s;
        // width eight accepts only three drafts but yields 4/0.5 = 8 t/s.
        for (drafts, accepted, seconds) in [(1, 1, 1.0), (3, 3, 4.0), (7, 3, 0.5)] {
            #expect(policy.nextDraftCount == drafts)
            policy.observe(drafted: drafts, accepted: accepted, elapsedSeconds: 1000)
            #expect(policy.nextDraftCount == drafts)
            policy.observe(drafted: drafts, accepted: accepted, elapsedSeconds: seconds)
        }
        #expect(policy.nextDraftCount == 7)
    }

    @Test("Caps are respected and budget-clipped tails do not train", arguments: [2, 3, 4, 6, 8, 64])
    func bounds(_ maximum: Int) {
        var policy = MTPAdaptiveDraftPolicy(maximumBlockSize: maximum)
        for _ in 0..<100 {
            let drafts = policy.nextDraftCount
            #expect((1..<min(maximum, 8)).contains(drafts))
            policy.observe(drafted: drafts - 1, accepted: 0, elapsedSeconds: 0.00001)
            #expect(policy.nextDraftCount == drafts)
            policy.observe(drafted: drafts, accepted: drafts, elapsedSeconds: 1)
        }
    }

    @Test("Invalid timings cannot poison throughput estimates")
    func invalidObservations() {
        var policy = MTPAdaptiveDraftPolicy(maximumBlockSize: 4)
        for elapsed in [0.0, -1.0, .nan, .infinity] {
            policy.observe(drafted: 1, accepted: 1, elapsedSeconds: elapsed)
        }
        policy.observe(drafted: 1, accepted: 2, elapsedSeconds: 1)
        policy.observe(drafted: 1, accepted: 1, elapsedSeconds: 1)
        #expect(policy.nextDraftCount == 1, "Invalid observations must not consume the warmup")
        policy.observe(drafted: 1, accepted: 1, elapsedSeconds: 1)
        #expect(policy.nextDraftCount == 3)
    }

    @Test("Periodic probes revisit widths as throughput changes")
    func adaptation() {
        var policy = MTPAdaptiveDraftPolicy(maximumBlockSize: 4)
        for _ in 0..<2 {
            policy.observe(drafted: 1, accepted: 1, elapsedSeconds: 1)
        }
        for _ in 0..<2 {
            policy.observe(drafted: 3, accepted: 3, elapsedSeconds: 4)
        }
        #expect(policy.nextDraftCount == 1)
        var revisitedWide = false
        for _ in 0..<40 {
            let drafts = policy.nextDraftCount
            if drafts == 3 {
                revisitedWide = true
            }
            policy.observe(drafted: drafts, accepted: drafts, elapsedSeconds: drafts == 3 ? 0.1 : 1)
        }
        #expect(revisitedWide)
        #expect(policy.nextDraftCount == 3)
    }
}
