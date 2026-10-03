import Foundation
import ModelRunnerProtocol
import loom
import weft

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

/// Owns console state, rendering and input on the main actor; the lifecycle actor owns model operations.
@MainActor
final class ServerConsole {
    private let manager: ModelLifecycleManager
    private let endpoint: String
    private var listening = false
    private var dashboard = ServerDashboardSnapshot()
    private let startedAt = ProcessInfo.processInfo.systemUptime
    private var state: ModelLifecycleState?
    private var models: [ModelLifecycleDescriptor] = []
    private var unavailableModels: Set<String> = []
    private var selectedIndex = 0
    private var message = "Starting listener…"
    private var actionMessage: String?
    private var terminal: Terminal<ConsoleBackend>?
    private var output: ConsoleOutput?
    private var management = ConsoleModelManagement()
    private var downloadTask: Task<Void, Never>?
    private var downloadProgress: Progress?
    private var downloadMessage: String?

    nonisolated static func shouldPresent(
        disabled: Bool, inputIsTerminal: Bool = isatty(STDIN_FILENO) == 1,
        outputIsTerminal: Bool = isatty(STDOUT_FILENO) == 1,
        terminalType: String? = ProcessInfo.processInfo.environment["TERM"]
    ) -> Bool {
        !disabled && inputIsTerminal && outputIsTerminal && terminalType != "dumb"
    }

    init(manager: ModelLifecycleManager, host: String, port: Int) {
        self.manager = manager
        let address = host.contains(":") ? "[\(host)]" : host
        endpoint = "http://\(address):\(port)/v1"
    }

    func didStartListening() {
        listening = true
        message = "Select a model and press Enter to load it."
    }

    func run() async throws {
        await manager.setGenerationMonitoringEnabled(true)
        let session = try TerminalSession(options: TerminalOptions(alternateScreen: true, hideCursor: true))
        defer {
            downloadTask?.cancel()
            session.close()
        }
        let capture = try ConsoleOutput()
        output = capture
        defer {
            terminal = nil
            output = nil
            capture.restore()
        }
        let input = TerminalEvents(session: session)
        defer { input.stop() }
        let size = capture.terminalSize
        terminal = Terminal(backend: capture.backend, width: size.width, height: size.height)
        try await refresh(reloadCatalog: true)

        do {
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask { try await self.consume(input) }
                group.addTask { try await self.refreshContinuously() }
                defer { group.cancelAll() }
                _ = try await group.next()
            }
        } catch {
            downloadTask?.cancel()
            await downloadTask?.value
            throw error
        }
        downloadTask?.cancel()
        await downloadTask?.value
    }

    private func consume(_ input: TerminalEvents) async throws {
        for try await batch in input.events {
            try Task.checkCancellation()
            for event in batch {
                if try await handle(event) {
                    return
                }
            }
            try draw()
        }
    }

    private func refreshContinuously() async throws {
        while !Task.isCancelled {
            try await Task.sleep(for: .milliseconds(250))
            try await refresh(reloadCatalog: false)
        }
    }

    private func refresh(reloadCatalog: Bool) async throws {
        if let size = output?.terminalSize {
            terminal?.resize(width: size.width, height: size.height)
        }
        let previousGeneration = state?.modelGeneration
        let previousPhase = state?.phase
        state = await manager.state()
        dashboard = await manager.dashboardSnapshot()
        dashboard.uptimeSeconds = ProcessInfo.processInfo.systemUptime - startedAt
        if listening, state?.phase != previousPhase {
            actionMessage = nil
            switch state?.phase {
            case .empty:
                message = "No model loaded. Select a model and press Enter."
            case .ready:
                message = "Model ready."
            case .loading:
                message = "Loading model…"
            case .draining:
                message = "Waiting for active requests…"
            case .unloading:
                message = "Releasing model memory…"
            case nil:
                break
            }
        }
        if reloadCatalog || state?.modelGeneration != previousGeneration {
            let selectedID = models.indices.contains(selectedIndex) ? models[selectedIndex].id : nil
            let entries = await manager.modelAvailability()
            models = entries.map(\.model)
            unavailableModels = Set(entries.filter { !$0.available }.map { $0.model.id })
            selectedIndex = models.firstIndex(where: { $0.id == selectedID }) ?? 0
        }
        try draw()
    }

    private func handle(_ event: InputEvent) async throws -> Bool {
        switch event {
        case .signal:
            return true
        case .resize:
            if let size = output?.terminalSize {
                terminal?.resize(width: size.width, height: size.height)
            }
        default:
            break
        }
        guard let keyboard = event.keyboardEvent, keyboard.kind != .release else {
            return false
        }
        if management.deletionTarget != nil {
            switch keyboard.key {
            case .character("y"):
                if let id = management.confirmDeletion() {
                    do {
                        _ = try await manager.removeDownload(named: id)
                        try await refresh(reloadCatalog: true)
                        actionMessage = "Deleted \(id) from disk."
                    } catch { actionMessage = error.localizedDescription }
                }
            case .escape, .character("n"):
                management.cancelDeletion()
                actionMessage = "Deletion cancelled."
            case .interrupt, .endOfInput: return true
            default: break
            }
            return false
        }
        if management.showingDownloads {
            switch keyboard.key {
            case .escape, .character("d"):
                management.showingDownloads = false
            case .up, .character("k"):
                management.downloadIndex = max(0, management.downloadIndex - 1)
            case .down, .character("j"):
                management.downloadIndex = min(ModelDownloadCatalog.entries.count - 1, management.downloadIndex + 1)
            case .enter: startDownload()
            case .character("c"): downloadTask?.cancel()
            case .character("t"):
                let present = await HuggingFaceTokenStatus.available()
                actionMessage =
                    present
                    ? "Hugging Face token available (not validated)."
                    : "No HF token. Use midnight auth login in another terminal."
            case .interrupt, .endOfInput, .character("q"): return true
            default: break
            }
            return false
        }
        switch keyboard.key {
        case .character("d"):
            management.showingDownloads = true
            actionMessage = "Publisher-owned ~4-bit sources. Enter downloads with designated assistant."
        case .character("c"):
            downloadTask?.cancel()
        case .character("x"):
            guard listening, models.indices.contains(selectedIndex) else { return false }
            let id = models[selectedIndex].id
            do {
                let downloads = try await manager.downloadedModels()
                guard let download = downloads.data.first(where: { $0.id == id }) else {
                    actionMessage = "Only managed downloads can be deleted."
                    return false
                }
                guard !download.inUse else {
                    actionMessage = "Unload this model before deleting it."
                    return false
                }
                management.requestDeletion(id)
            } catch { actionMessage = error.localizedDescription }
        case .interrupt, .endOfInput, .escape, .character("q"):
            return true
        case .up, .character("k"):
            selectedIndex = max(0, selectedIndex - 1)
        case .down, .character("j"):
            selectedIndex = min(max(0, models.count - 1), selectedIndex + 1)
        case .character("r"):
            actionMessage = nil
            try await refresh(reloadCatalog: true)
        case .character("a"):
            guard listening, models.indices.contains(selectedIndex) else {
                return false
            }
            let id = models[selectedIndex].id
            let available = unavailableModels.contains(id)
            do {
                state = try await manager.setModelAvailable(available, id: id)
                actionMessage = "\(id) is now \(available ? "available" : "unavailable")."
                try await refresh(reloadCatalog: true)
            } catch {
                actionMessage = error.localizedDescription
            }
        case .enter, .character("l"):
            guard listening, models.indices.contains(selectedIndex), let request = models[selectedIndex].loadRequest
            else {
                return false
            }
            guard !unavailableModels.contains(models[selectedIndex].id) else {
                actionMessage = "Press a to make this model available before loading it."
                return false
            }
            do {
                state = try await manager.load(request)
                actionMessage = nil
                message = "Loading \(models[selectedIndex].id)…"
            } catch {
                actionMessage = error.localizedDescription
            }
        case .character("u"):
            guard listening else {
                return false
            }
            do {
                state = try await manager.unload(
                    expectedGeneration: state?.modelGeneration, expectedInstanceID: state?.instanceID)
                actionMessage = nil
                message = state?.phase == .empty ? "No model loaded." : "Unloading model…"
            } catch {
                actionMessage = error.localizedDescription
            }
        default:
            break
        }
        return false
    }

    private func startDownload() {
        guard downloadTask == nil else {
            actionMessage = "A download is already running. Press c to cancel."
            return
        }
        let entry = ModelDownloadCatalog.entries[management.downloadIndex]
        if let reason = entry.unavailableReason {
            actionMessage = reason
            return
        }
        downloadMessage = "Checking \(entry.name)…"
        downloadTask = Task { [weak self] in
            do {
                try await ModelDownloadService.download(repository: entry.repository, name: entry.name) {
                    [weak self] message, progress in
                    self?.downloadMessage = message
                    self?.downloadProgress = progress
                }
                try await self?.refresh(reloadCatalog: true)
                self?.actionMessage = "Downloaded \(entry.name)."
            } catch is CancellationError {
                self?.actionMessage = "Download cancelled; reusable Hub cache is retained."
            } catch {
                self?.actionMessage = ModelDownloadService.errorMessage(error)
            }
            self?.downloadTask = nil
            self?.downloadProgress = nil
            self?.downloadMessage = nil
        }
    }

    private func draw() throws {
        let logs = output?.lines() ?? []
        try terminal?.draw { frame in
            Self.render(
                frame: &frame, endpoint: endpoint, listening: listening, state: state,
                models: models, selectedIndex: selectedIndex, message: message, logs: logs,
                unavailableModels: unavailableModels,
                actionMessage: management.prompt ?? actionMessage,
                dashboard: dashboard,
                downloadEntries: management.showingDownloads ? ModelDownloadCatalog.entries : nil,
                downloadIndex: management.downloadIndex,
                downloadStatus: downloadMessage.map { text in
                    if let progress = downloadProgress { return "\(text) · \(Int(progress.fractionCompleted * 100))%" }
                    return text
                })
        }
    }

    /// Layout uses terminal cells and clips safely when resized below the usual console dimensions.
    static func render(
        frame: inout Frame, endpoint: String, listening: Bool, state: ModelLifecycleState?,
        models: [ModelLifecycleDescriptor], selectedIndex: Int, message: String, logs: [String],
        unavailableModels: Set<String> = [], actionMessage: String? = nil,
        dashboard: ServerDashboardSnapshot = ServerDashboardSnapshot(),
        downloadEntries: [ModelDownloadCatalog.Entry]? = nil, downloadIndex: Int = 0,
        downloadStatus: String? = nil
    ) {
        let width = max(1, frame.area.width - 2)
        func row(_ text: String, at y: Int, style: Style = Style()) -> (Paragraph, Rect) {
            (Paragraph(TextLayout(text, width: width), style: style), Rect(x: 1, y: y, width: width, height: 1))
        }
        func drawRow(_ text: String, at y: Int, style: Style = Style()) {
            let (paragraph, area) = row(text, at: y, style: style)
            frame.render(paragraph, in: area)
        }
        drawRow("midnight", at: 0, style: Style(foreground: .indexed(6), bold: true))
        drawRow(endpoint, at: 1)
        let phase = listening ? (state?.phase.rawValue ?? "empty") : "starting"
        let model = state?.loadedModel?.id ?? state?.targetModel ?? "No model loaded"
        let activity = state?.phase == .loading ? " \(ServerDashboard.spinner(dashboard.uptimeSeconds))" : ""
        drawRow("\(phase.uppercased())\(activity)  ·  \(model)", at: 3, style: Style(bold: true))
        ServerDashboard.render(frame: &frame, state: state, snapshot: dashboard)
        let expanded = frame.area.height >= 20
        let menuStart = 14
        if expanded {
            drawRow(
                downloadEntries == nil ? "Installed models" : "Publisher downloads · HF token: t", at: menuStart - 1,
                style: Style(foreground: .indexed(6), bold: true))
        }
        let menuHeight = expanded ? max(0, frame.area.height - 20) : 0
        let index = downloadEntries == nil ? selectedIndex : downloadIndex
        let start = max(0, index - max(0, menuHeight - 1))
        let installedItems = models.dropFirst(start).map {
            let unavailable = unavailableModels.contains($0.id)
            let hint = unavailable ? "unavailable" : (state?.loadedModel?.id == $0.id ? "loaded" : "available")
            return MenuItem("\(unavailable ? "−" : "+") \($0.id)", hint: hint)
        }
        let items =
            downloadEntries.map { entries in
                entries.dropFirst(start).map {
                    MenuItem($0.name, hint: $0.unavailableReason == nil ? "download" : "unavailable")
                }
            } ?? Array(installedItems)
        frame.render(
            Menu(items, selectedIndex: index - start),
            in: Rect(x: 1, y: menuStart, width: width, height: menuHeight))
        if expanded {
            if models.isEmpty && downloadEntries == nil {
                drawRow("No installed models. Press d to download.", at: menuStart)
            }
            let logStart = frame.area.height - 6
            drawRow(
                actionMessage ?? downloadStatus ?? state?.lastError ?? message, at: logStart,
                style: Style(foreground: .indexed(3)))
            if let downloadStatus {
                drawRow(downloadStatus, at: logStart + 1, style: Style(foreground: .indexed(6)))
            } else if let downloadEntries, downloadEntries.indices.contains(downloadIndex) {
                drawRow(downloadEntries[downloadIndex].note, at: logStart + 1)
            }
            for (index, line) in logs.suffix(3).enumerated() {
                drawRow(line, at: logStart + 2 + index, style: Style(foreground: .indexed(8)))
            }
        }
        drawRow(
            downloadEntries == nil
                ? "↑/↓ · Enter load · a availability · u unload · d download · x delete · r refresh · q quit"
                : "↑/↓ select · Enter download · c cancel · t token status · Esc back · q quit",
            at: frame.area.height - 1)
    }
}
