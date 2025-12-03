# Authentication API Documentation

This document describes the authentication APIs for the Web Chatbot System. These APIs allow users to register, login, and manage their chatbot instances in a multi-tenant plugin model.

## Table of Contents

- [Overview](#overview)
- [Authentication Flow](#authentication-flow)
- [API Endpoints](#api-endpoints)
  - [Register User](#register-user)
  - [Login](#login)
  - [Get Current User](#get-current-user)
  - [Update Website](#update-website)
- [Using JWT Tokens](#using-jwt-tokens)
- [Error Handling](#error-handling)
- [Example Usage](#example-usage)

## Overview

The authentication system uses:
- **JWT (JSON Web Tokens)** for stateless authentication
- **Bcrypt** for secure password hashing
- **Bearer token** authentication for protected endpoints

Each registered user gets:
- A unique user ID
- An API key for plugin integration
- The ability to configure their own website URL
- Secure authentication with JWT tokens (valid for 7 days)

## Authentication Flow

1. **Register** a new account using `/register`
2. **Login** with email and password to get a JWT token using `/login`
3. **Use the token** in the `Authorization` header for protected endpoints
4. **Token expires** after 7 days - login again to get a new token

## API Endpoints

### Register User

Register a new user account.

**Endpoint:** `POST /register`

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123",
  "organization_name": "My Company",
  "full_name": "John Doe"
}
```

**Required Fields:**
- `email`: User email address (used as username)
- `password`: User password (minimum 8 characters)

**Optional Fields:**
- `organization_name`: Name of the organization/company
- `full_name`: User's full name

**Response (201 Created):**
```json
{
  "status": "success",
  "message": "User registered successfully",
  "user": {
    "user_id": "abc123def456",
    "email": "user@example.com",
    "organization_name": "My Company",
    "full_name": "John Doe",
    "created_at": "2024-01-15T10:30:00",
    "is_active": true,
    "website_url": "",
    "api_key": "xyz789abc123..."
  },
  "note": "Save your API key securely - you'll need it for plugin integration"
}
```

**Error Responses:**
- `400 Bad Request`: User already exists or password too short
- `500 Internal Server Error`: Server error during registration

**Example (cURL):**
```bash
curl -X POST "http://localhost:5000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123",
    "organization_name": "My Company",
    "full_name": "John Doe"
  }'
```

---

### Login

Login and get a JWT access token.

**Endpoint:** `POST /login`

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Login successful",
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 604800,
  "user": {
    "user_id": "abc123def456",
    "email": "user@example.com",
    "organization_name": "My Company",
    "full_name": "John Doe",
    "created_at": "2024-01-15T10:30:00",
    "is_active": true,
    "website_url": "",
    "api_key": "xyz789abc123..."
  }
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid email or password
- `500 Internal Server Error`: Server error during login

**Example (cURL):**
```bash
curl -X POST "http://localhost:5000/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123"
  }'
```

---

### Get Current User

Get information about the currently authenticated user.

**Endpoint:** `GET /me`

**Authentication:** Required (Bearer token)

**Headers:**
```
Authorization: Bearer <your_jwt_token>
```

**Response (200 OK):**
```json
{
  "status": "success",
  "user": {
    "user_id": "abc123def456",
    "email": "user@example.com",
    "organization_name": "My Company",
    "full_name": "John Doe",
    "created_at": "2024-01-15T10:30:00",
    "is_active": true,
    "website_url": "https://example.com",
    "api_key": "xyz789abc123..."
  }
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid or expired token

**Example (cURL):**
```bash
curl -X GET "http://localhost:5000/me" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

---

### Update Website

Update the website URL associated with the user's account.

**Endpoint:** `POST /update_website`

**Authentication:** Required (Bearer token)

**Headers:**
```
Authorization: Bearer <your_jwt_token>
```

**Request Body (Form Data or JSON):**
```json
{
  "website": "https://example.com"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Website URL updated successfully",
  "user": {
    "user_id": "abc123def456",
    "email": "user@example.com",
    "organization_name": "My Company",
    "full_name": "John Doe",
    "created_at": "2024-01-15T10:30:00",
    "is_active": true,
    "website_url": "https://example.com",
    "api_key": "xyz789abc123..."
  }
}
```

**Error Responses:**
- `400 Bad Request`: Website URL is required
- `401 Unauthorized`: Invalid or expired token
- `500 Internal Server Error`: Server error

**Example (cURL):**
```bash
curl -X POST "http://localhost:5000/update_website" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "website": "https://example.com"
  }'
```

## Using JWT Tokens

After logging in, you'll receive a JWT token. Use this token to authenticate protected endpoints:

1. **Include the token** in the `Authorization` header:
   ```
   Authorization: Bearer <your_jwt_token>
   ```

2. **Token expiration**: Tokens are valid for 7 days. After expiration, login again to get a new token.

3. **Token format**: The token is a long string that looks like:
   ```
   eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyQGV4YW1wbGUuY29tIiwiZXhwIjoxNzA1MzI0MDAwfQ...
   ```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "status": "error",
  "message": "Error description here"
}
```

Common HTTP status codes:
- `200 OK`: Success
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required or invalid
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## Example Usage

### Complete Flow Example (Python)

```python
import requests

BASE_URL = "http://localhost:5000"

# 1. Register a new user
register_data = {
    "email": "user@example.com",
    "password": "securepassword123",
    "organization_name": "My Company",
    "full_name": "John Doe"
}

response = requests.post(f"{BASE_URL}/register", json=register_data)
if response.status_code == 201:
    user_data = response.json()
    print(f"Registered! API Key: {user_data['user']['api_key']}")
else:
    print(f"Registration failed: {response.json()}")

# 2. Login
login_data = {
    "email": "user@example.com",
    "password": "securepassword123"
}

response = requests.post(f"{BASE_URL}/login", json=login_data)
if response.status_code == 200:
    token_data = response.json()
    access_token = token_data["access_token"]
    print(f"Logged in! Token: {access_token[:50]}...")
else:
    print(f"Login failed: {response.json()}")

# 3. Get current user info
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get(f"{BASE_URL}/me", headers=headers)
if response.status_code == 200:
    user_info = response.json()
    print(f"Current user: {user_info['user']['email']}")

# 4. Update website
website_data = {"website": "https://example.com"}
response = requests.post(
    f"{BASE_URL}/update_website",
    json=website_data,
    headers=headers
)
if response.status_code == 200:
    print("Website updated successfully!")
```

### JavaScript/TypeScript Example

```javascript
const BASE_URL = 'http://localhost:5000';

// 1. Register
async function register(email, password, organizationName, fullName) {
  const response = await fetch(`${BASE_URL}/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email,
      password,
      organization_name: organizationName,
      full_name: fullName
    })
  });
  return await response.json();
}

// 2. Login
async function login(email, password) {
  const response = await fetch(`${BASE_URL}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  });
  return await response.json();
}

// 3. Get current user (with token)
async function getCurrentUser(token) {
  const response = await fetch(`${BASE_URL}/me`, {
    headers: { 'Authorization': `Bearer ${token}` }
  });
  return await response.json();
}

// Usage
(async () => {
  // Register
  const registerResult = await register(
    'user@example.com',
    'securepassword123',
    'My Company',
    'John Doe'
  );
  console.log('Registered:', registerResult);

  // Login
  const loginResult = await login('user@example.com', 'securepassword123');
  const token = loginResult.access_token;
  console.log('Logged in, token:', token);

  // Get user info
  const userInfo = await getCurrentUser(token);
  console.log('Current user:', userInfo);
})();
```

## Security Notes

1. **Password Requirements**: Minimum 8 characters
2. **Token Storage**: Store JWT tokens securely (e.g., in memory, secure cookies, or encrypted storage)
3. **HTTPS**: In production, always use HTTPS to protect tokens in transit
4. **API Key**: Each user gets a unique API key for plugin integration - keep it secure
5. **Token Expiration**: Tokens expire after 7 days - implement token refresh logic if needed

## Integration with Plugin Model

For plugin integration, you can use either:
1. **JWT Token**: For authenticated API calls
2. **API Key**: For programmatic access (stored in user account)

Both can be used to identify and authenticate requests from your plugin installations.

