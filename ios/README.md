# ShelfAware iPhone app — 6.1.0 (8)

This is the selected build 8 source. Generate the project with XcodeGen; generated projects and personal signing settings are ignored by Git.

Use the full Xcode toolchain compatible with Meta DAT 1.0.1 (the existing environment checks target Xcode 26.4+ / Swift 6.3+) and XcodeGen. The app's deployment target is iOS 17.2.

## Targets

| Specification | Scheme | Capture path |
| --- | --- | --- |
| `project-personal.yml` | `ShelfAwarePersonal` | Experimental Bluetooth Classic glasses transport |
| `project.yml` | `ShelfAware` | Original glasses integration with configured capabilities |
| `project-phone.yml` | `ShelfAwarePhone` | Explicit phone-camera fallback without Meta DAT |

For the selected Bluetooth target, run from this directory:

```sh
xcodegen generate --spec project-personal.yml
mkdir -p ShelfAwarePersonal.xcodeproj/project.xcworkspace/xcshareddata/swiftpm
cp Package.resolved ShelfAwarePersonal.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved
open ShelfAwarePersonal.xcodeproj
```

`Package.resolved` records Meta DAT **1.0.1**, revision `0bd369298eef6a980ad0bdb6c8a5502f2c02a08c`. The YAML specifications also require that exact version. The SDK downloads separately and is subject to its own terms.

Choose your signing team and unique bundle identifier in Xcode, and configure the appropriate Meta developer app values when required. `META_APP_ID` and `CLIENT_TOKEN` are placeholders. Keep real values in local configuration. A connected, trusted iPhone, device Developer Mode, compatible paired glasses, and Meta developer setup are required for glasses trials. The phone target does not prove glasses functionality.

Start the backend using the [root instructions](../README.md). In the app, enter its reachable LAN origin and private bearer token, then check the connection. The token remains in memory and must be entered again after relaunch. Provider API keys stay on the backend. Foreground conversation requires explicit consent; microphone capture pauses while processing and playing a reply.

To verify the phone target without signing:

```sh
xcodegen generate --spec project-phone.yml
xcodebuild -project ShelfAwarePhone.xcodeproj -scheme ShelfAwarePhone -destination 'generic/platform=iOS' -derivedDataPath DerivedData/Phone CODE_SIGNING_ALLOWED=NO build
```

The source-only scene stability checks are described in the root validation commands. Local builds and tests do not establish current glasses capture, sound routing, or model accuracy. See [native third-party notices](THIRD_PARTY_NOTICES.md).
