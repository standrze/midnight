# Terminal interface architecture review

Reviewed OnoSendai (`/Users/stephen/asslayer`) and Midnight Studio
(`/Users/stephen/Documents/ChatGPT/midnight-studio`) against the shared loom/weft
application pattern. This is an architecture and lifecycle review, not a complete
feature, security, or cross-platform audit.

| Responsibility | OnoSendai | Midnight Studio |
| --- | --- | --- |
| Shared dependencies | SwiftPM loom/weft, Swift 6.4 | SwiftPM loom/weft, Swift 6.4 |
| Launch policy | ArgumentParser entrypoint; interactive or plain mode | Launch invocation precedes terminal setup; terminal and JSON commands; explicit legacy web command |
| Terminal lifecycle | PromptApplication owns a weft TerminalSession and TerminalEvents, closed/stopped with defer | StudioTerminal owns a weft TerminalSession and TerminalEvents, closed/stopped with defer |
| Rendering | loom Terminal, Frame, widgets, WorkspaceShell and application views | loom Terminal, Frame and Paragraph; viewport clipping and control-character filtering |
| UI state | Main-actor PromptApplication and application state | Main-actor StudioTerminal |
| Model/application work | HarnessCore, standalone OpenAICompatible client, plugin SDK and application services | ModelControlManager actor, catalog, HTTP client and owned runtime process |
| Backend ownership | App owns policy and UI; shared libraries do not import the application | Runtime serving stays in Midnight; new model development belongs in Afterglow |

Both follow the shared architecture. Their screen complexity and branding can
remain different; using the same terminal libraries does not require identical
forms or navigation.

## Changes made

OnoSendai previously used separate teardown implementations for requested quit,
input exhaustion, and errors. PromptApplication now calls the same application
shutdown function on every main event-loop exit path, including browser/plugin and owned task cleanup
before terminal restoration. The terminal-wrap image test's mutating operation
is evaluated before passing its result to the Swift Testing requirement macro.

Studio now separates inspection transport and response validation into
StudioInspectionClient. It rejects an in-flight result when the manager's routing
revision changes and rejects cancelled requests. StudioTerminal also clears old
inspection rows and cancels the request when its model connection changes.
Tests cover stable, changed, and cancelled inspection requests.

## Verification scope

Studio's 49 Swift tests, touched-file format checks, and PTY checks passed.
The PTY checks include resizing, quit, Ctrl-C, alternate-screen restoration,
terminal attribute restoration, and JSON output without terminal escapes.

OnoSendai's current quit-binding PTY checks passed in Chat and Dashboard. The broad
catalog smoke passed command handling and tiny-resize recovery, then stopped on
its historical lowercase settings label (`model endpoint`; current UI uses
`Model endpoint`). This is a fixture mismatch, not proof that the remaining mouse
and settings checks pass. Project-wide formatting reported issues in unrelated
AgentDashboard, AgentRecord, and AgentCoordinatorTests files; the changed Swift
files pass formatting. The full Swift test build failed in the unrelated
`CodeSyntaxTests.swift:42`: the parameter is named `Theme`, while the body
references `theme`. The full suite therefore remains unverified. This checkout
was also being edited and built by another chat during validation.

Linux terminal behavior and native plugin teardown on forced process termination
were not validated in this review.
