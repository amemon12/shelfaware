# Native bridge dependencies and provenance

The app's shopping/session/UI/HTTP code is new work for Shelf Aware. The small DAT adapter follows the official documented lifecycle and sample API usage; it is not a fork of VisionClaw, Omnivision, OpenGlasses, or TurboMeta.

**Meta Wearables Device Access Toolkit for iOS, exact version 1.0.1.** Vendor: Meta Platforms, Inc. and affiliates. Its SDK binaries and official sample/documentation are governed by Meta's applicable terms, not by an assumed MIT or Apache license. Developers must read and accept the applicable terms before use. The official sample header says: “Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.”

- Repository/license: https://github.com/facebook/meta-wearables-dat-ios/blob/main/LICENSE
- Source patterns examined: https://github.com/facebook/meta-wearables-dat-ios/tree/bbae597cd1c748cd29b9375a1624a8d75b1c9e14/samples/CameraAccess
- Adapted API/lifecycle pattern location in this app: `ShelfAware/GlassesCamera.swift`.
- Configuration based on official sample/guide: `ShelfAware/Info.plist`, `ShelfAware/ShelfAware.entitlements`, exact package requirement in `project.yml`.

**Apple frameworks:** SwiftUI, UIKit, Foundation, CoreGraphics, Combine, AVFoundation, Speech, Vision, and ImageIO are system frameworks, subject to Apple's platform/developer terms. Their inclusion is not proof of any entitlement eligibility.

**XcodeGen** is a project generation tool maintained at https://github.com/yonaskolb/XcodeGen under its own MIT license. It is not bundled into the application and is not invoked automatically by this repository.

Backend providers are accessed through the Shelf Aware API; no provider keys or SDK binaries are bundled here. See the root third-party notices for backend and research provenance.
