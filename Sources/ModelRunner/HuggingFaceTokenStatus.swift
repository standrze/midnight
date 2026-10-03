import HuggingFace

enum HuggingFaceTokenStatus {
    /// Only report presence; never return or display credentials in the console.
    static func available() async -> Bool {
        await HubClient().bearerToken?.isEmpty == false
    }
}
