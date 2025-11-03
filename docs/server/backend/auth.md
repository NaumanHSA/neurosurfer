---
title: Auth & Users
description: JWT authentication with bcrypt passwords, secure cookies, sliding refresh sessions, and easy FastAPI dependencies for Neurosurfer.
---

# Auth & Users

Neurosurfer ships with a pragmatic auth layer that works for both **browser apps** and **API clients**. Passwords are stored with **bcrypt**; authenticated sessions use **JWT access tokens** that can live either in the `Authorization: Bearer` header or in a secure **HttpOnly cookie**. Sessions automatically extend using a **sliding refresh** so active users stay signed in without surprise logouts.

---

## Capabilities at a Glance

- Password hashing & verification with **bcrypt**  
- Short‑lived **JWT** access tokens (`HS256`), created and verified server‑side  
- **Two auth modes**: `Authorization: Bearer <token>` **or** secure **cookie**  
- **Sliding session refresh** that extends tokens on activity  
- Drop‑in dependencies: ``get_current_user`` (required) and ``maybe_current_user`` (optional)

---

## Token & Cookie Model

A session is a signed JWT containing the user id in ``sub`` and standard ``iat`` / ``exp`` timestamps. You can send the token either:

- In the **Authorization** header: ``Authorization: Bearer <token>``  
- In an **HttpOnly cookie** (server sets it via ``set_login_cookie``)

For browser apps, the cookie path keeps tokens out of JavaScript by default. Cookies are configured with `Secure`, `HttpOnly`, and `SameSite` using your app config.

**Sliding refresh.** On each authenticated request, if the token is close to expiry (e.g., less than 60 minutes remaining), the server **rotates** it and resets the cookie, keeping the session alive while the user is active.

---

## Core Dependencies

### `get_current_user` — required auth

Use this when a route must be authenticated. It retrieves the token (header first, then cookie), validates it, fetches the user from the database, **slides** the session if needed, and makes the user available at both the return value and `request.state.user`.

```python
from fastapi import Depends

@app.endpoint("/me", method="get", dependencies=[])
def me(user = Depends(get_current_user)):
    return {"id": user.id, "email": user.email}
```

If the token is missing/invalid, the dependency raises `401` with a compact error message.

### `maybe_current_user` — soft auth

For routes that should work with or without a logged‑in user, use the best‑effort resolver. It returns a `User | None` and never raises; invalid or missing tokens are treated as anonymous access.

```python
@app.endpoint("/whoami", method="get")
def whoami(user = Depends(maybe_current_user)):
    return {"user": getattr(user, "email", None)}
```

---

## Login & Logout (Sketch)

A typical login flow verifies credentials, issues a token, and sets the cookie:

```python
from fastapi import HTTPException, Response

@app.endpoint("/login", method="post", request=LoginIn, response=LoginOut)
def login(data: LoginIn, db = Depends(get_db), response: Response = None):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.id})
    set_login_cookie(response, token)
    return LoginOut(ok=True)
```

To log out, simply clear the cookie:

```python
@app.endpoint("/logout", method="post")
def logout(response: Response):
    clear_login_cookie(response)
    return {"ok": True}
```

---

## Using Auth in Chat Handlers

If you want to bind generation to the current user (for example to **scope RAG** or **quota**), pull the user from the request state (populated by the dependency), or include `maybe_current_user` as a dependency on the router that mounts your handler.

```python
@app.chat()
def handle_chat(request, ctx):
    user = getattr(ctx.request.state, "user", None)
    actor_id = resolve_actor_id(ctx.request, user)
    # use actor_id to scope threads, logs, and RAG collections
    ...
```

`resolve_actor_id` picks a stable integer from (in order): the authenticated user, the `X-Actor-Id` header, `?aid=` query param, or `0` (anonymous). This makes it easy to correlate logs and data even for non‑authenticated traffic.

---

## cURL Examples

### Header bearer token

```bash
curl -X GET http://localhost:8000/v1/models   -H "Authorization: Bearer <TOKEN>"
```

### Cookie session (browser‑style)

```bash
# after a login that set the cookie
curl -X GET http://localhost:8000/v1/chats   --cookie "nm_session=<TOKEN>"
```

> Replace the cookie name with the one configured in your app. Cookies are `HttpOnly` and typically set with `Secure` and `SameSite` based on environment.

---

## Security Notes

Keep sessions short‑lived and rotate keys periodically. Ensure `Secure` cookies in production, enable CORS for your frontend origin, and avoid putting tokens in local storage. The sliding refresh threshold is configurable; pick a value that balances UX and security. For high‑risk routes, layer rate limits and device‑level checks as **dependencies**.

---