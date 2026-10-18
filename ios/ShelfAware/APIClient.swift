import Foundation

/// A per-session client. Provider keys never leave the backend.
struct APIClient {
    private struct ErrorBody: Decodable { let detail: String }

    let baseURL: URL
    let bearerToken: String
    private static let transport: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: configuration)
    }()

    init(address: String, bearerToken: String) throws {
        guard let url = URL(string: address.trimmingCharacters(in: .whitespacesAndNewlines)),
              ["http", "https"].contains(url.scheme?.lowercased() ?? ""),
              url.host != nil, url.user == nil, url.password == nil,
              url.query == nil, url.fragment == nil, ["", "/"].contains(url.path) else {
            throw AppError.message("Enter the backend origin, for example http://192.168.1.20:8000. Use the computer's LAN address, not localhost.")
        }
        baseURL = url
        self.bearerToken = bearerToken.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func request<Response: Decodable>(_ path: String, method: String = "GET", body: Data? = nil) async throws -> Response {
        let data = try await responseData(path, method: method, body: body)
        return try JSONDecoder().decode(Response.self, from: data)
    }

    /// The health endpoint is public. Probe an existing protected read-only route
    /// without creating a session or using its fixture data for a live capture.
    func checkAuthorization() async throws {
        _ = try await responseData("fixtures")
    }

    private func responseData(_ path: String, method: String = "GET", body: Data? = nil) async throws -> Data {
        var request = URLRequest(url: baseURL.appendingPathComponent("api").appendingPathComponent(path))
        request.httpMethod = method
        request.timeoutInterval = 45
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        if !bearerToken.isEmpty { request.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization") }
        let (data, response) = try await Self.transport.data(for: request)
        try Task.checkCancellation()
        guard let http = response as? HTTPURLResponse else { throw AppError.message("The backend returned no HTTP response.") }
        if http.statusCode == 401 {
            throw AppError.message("API authorization required. Paste the current LAN bearer token in Connection, then tap Check backend. The token must be entered again after the app relaunches.")
        }
        guard (200..<300).contains(http.statusCode) else {
            let detail = (try? JSONDecoder().decode(ErrorBody.self, from: data).detail) ?? "Request failed (HTTP \(http.statusCode))."
            throw AppError.message(detail)
        }
        return data
    }

    func post<Body: Encodable, Response: Decodable>(_ path: String, body: Body) async throws -> Response {
        try await request(path, method: "POST", body: JSONEncoder().encode(body))
    }

    func cancel(_ id: String) async throws -> ShoppingSession {
        try await request("sessions/\(id)/cancel", method: "POST")
    }
}
