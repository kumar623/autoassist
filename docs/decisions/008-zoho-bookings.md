# 008 - Bookings live in Zoho Bookings, reached over MCP

## Status
Accepted (week 4). Live since 20 September 2026.

## Context
Bookings were a JSON file inside the container. Known limitations: they were
lost on every deploy, each replica had its own copy, the workshop could not see
them, and the customer got no confirmation. Azure Table Storage was the planned
fix, and it solves only the first two.

The workshop uses **Zoho Bookings**, which already has the calendar, the staff
hours and the customer emails. Zoho publishes an **MCP server** for it.

## Decision
`BOOKING_BACKEND=zoho` puts bookings in Zoho Bookings through that MCP server.
`services/orchestrator/zoho_bookings.py` offers the same functions as
`booking.py`, so the tool layer, the prompts and every rule above them are
unchanged. The file backend stays as the default and as the offline test double.

## Why MCP here, when the rest of the service calls APIs directly
Decision 007 removed SDKs in favour of plain API calls, and MCP is a layer above
an API, so this looks like a contradiction. It is not:

- **The vendor publishes the MCP server.** We are not adding a layer; we are
  using the interface Zoho offers. The alternative is Zoho's REST API plus its
  own auth, which is more work, not less.
- **The orchestrator stays the client.** The agents do not call Zoho. They call
  our function tools, our code calls Zoho, and every call still appears in the
  trace with its arguments and timing. Tools the model can reach are still ours
  to guard.
- **It is still plain HTTPS.** `mcp_client.py` speaks MCP's JSON-RPC directly,
  with no MCP SDK - which also avoids the SDK's Python 3.10 requirement.

## Sign-in
Zoho's MCP URL contains an access key and is refused on its own (HTTP 401). The
server requires OAuth 2.1, advertised at the standard discovery addresses:
dynamic client registration, browser approval with PKCE, then refresh tokens.
`scripts/zoho_login.py` does this once; the tokens live in `.env` locally and in
Key Vault in Azure, where the container app holds references to them (decision
010). No password reaches the app.

## What Zoho gives us, and what it costs
Gained: appointments in the workshop's real calendar, surviving deploys, shared
by every replica, visible to staff, with Zoho emailing the customer on booking,
reschedule and cancellation. The confirmation email we were about to build is
Zoho's job, and Zoho is better at it.

Given up, and each one measured against the live account:
- **Zoho refuses a booking without a phone number**, though its own form marks
  Contact Number optional. The agent now collects name, email and phone, and
  `book_service_slot` refuses without them - in code, because the prompt alone
  did not hold.
- **Zoho has no field for a vehicle registration.** It went in the notes at
  first, and `rescheduleAppointment` WIPES the notes: a moved booking belonged
  to nobody, not even its owner. A custom booking field ("Vehicle Registration",
  single line) survives a reschedule and is what the code now uses
  (`ZOHO_REGISTRATION_FIELD`). Without it, moving is refused rather than
  silently breaking ownership.
- **Availability is Zoho's**, on its own 15-minute grid and staff hours, so the
  workshop's opening hours now live in Zoho rather than in `booking.py`.
- **An outage is now visible to customers.** Slot lookups and bookings fail with
  "the booking calendar could not be reached" when Zoho is down, where the file
  backend could not fail that way.
- **Customer data leaves Azure.** Names, emails and phone numbers are stored in
  Zoho, which is the point - the workshop needs them - but it is a second
  processor of personal data, in the EU region.

## Known limitation
If Zoho ever rotates the refresh token, the running container keeps the new one
only in memory: it cannot write back to a container app secret. A restart would
fall back to the stored one. Rotation was not seen in testing; the fix, if it
starts happening, is to store the token where the app can write - Key Vault or
Table Storage.

(21 September 2026: the token is now in Key Vault, but the app's identity has
only Key Vault Secrets User - read, not write - so this limitation stands. Fixing
it would mean letting the app write that one secret.)

## Revisit when
- Bookings need to be read back in bulk (reporting): Zoho's filters are limited,
  and a local index would help.
- A second workshop or workspace is added: service and staff ids are single
  values in configuration today.
