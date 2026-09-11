import Testing
@testable import ModelRunnerCore

@Suite struct GenerationAdmissionTests {
  @Test func queuesAndHandsOff() async throws {
    let gate = GenerationAdmission(capacity: 1)
    try await gate.acquire()
    let waiter = Task { try await gate.acquire(); gate.release() }
    while gate.queuedCount == 0 { await Task.yield() }
    do { try await gate.acquire(); Issue.record("Queue overflow admitted") }
    catch GenerationAdmission.Failure.full {}
    gate.release()
    try await waiter.value
    try await gate.acquire()
    gate.release()
  }
  @Test func cancellationRemovesWaiter() async throws {
    let gate = GenerationAdmission()
    try await gate.acquire()
    let waiter = Task { try await gate.acquire(); defer { gate.release() }; try Task.checkCancellation() }
    while gate.queuedCount == 0 { await Task.yield() }
    waiter.cancel()
    do { try await waiter.value; Issue.record("Cancellation ignored") } catch is CancellationError {}
    #expect(gate.queuedCount == 0)
    gate.release()
    try await gate.acquire()
    gate.release()
  }
  private actor Recorder {
    var values: [Int] = []
    func append(_ value: Int) { values.append(value) }
  }
  @Test func queuedRequestsRunInArrivalOrder() async throws {
    let gate = GenerationAdmission()
    let recorder = Recorder()
    try await gate.acquire()
    var tasks: [Task<Void, Error>] = []
    for index in 0..<8 {
      tasks.append(Task {
        try await gate.acquire()
        defer { gate.release() }
        await recorder.append(index)
      })
      while gate.queuedCount != index + 1 { await Task.yield() }
    }
    gate.release()
    for task in tasks { try await task.value }
    #expect(await recorder.values == Array(0..<8))
  }
  @Test func cancellationRacingHandoffDoesNotLeakOwnership() async throws {
    let gate = GenerationAdmission()
    for _ in 0..<100 {
      try await gate.acquire()
      let waiter = Task {
        try await gate.acquire()
        defer { gate.release() }
        try Task.checkCancellation()
      }
      while gate.queuedCount == 0 { await Task.yield() }
      waiter.cancel()
      gate.release()
      do { try await waiter.value } catch is CancellationError {}
    }
    try await gate.acquire()
    gate.release()
  }

}
