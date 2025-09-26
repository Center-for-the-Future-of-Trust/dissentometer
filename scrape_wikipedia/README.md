--concurrency 16

Global limit: the maximum number of HTTP requests the scraper will have “in flight” at once across all wikis.

In plain terms: it can talk to up to 16 different Wikipedia servers/pages simultaneously.

Higher = faster, but risks hitting API rate limits or stressing your network.

--per-host 5

Per-host limit: the maximum requests allowed at once to a single Wikipedia domain (like en.wikipedia.org).

So with --concurrency 16 --per-host 5:

Total across all languages: up to 16 requests at once.

For a single wiki (say English), no more than 5 at the same time.

This prevents hammering one language’s API server.

--batch-size 60

How many page titles to request at once when fetching extracts + langlinks in batch mode.

Instead of making one API request per page, the scraper groups them:

With batch size 60, it asks for up to 60 articles at once in a single API call.

Bigger batch = fewer total HTTP requests = more efficient.

But too big = slower responses, heavier JSON payloads, and higher memory use.
