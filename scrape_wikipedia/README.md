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

-------------

So if you don’t specify --langs, it runs with --langs all by default.

That means for every article in your root categories, the scraper will:

Save the English version (or whatever the base language is).

Use the langlinks API to discover all interlanguage links for that article.

Save every variant (French, Spanish, German, Japanese, etc.) if a link exists.

⚠️ A nuance:

“All” here means all languages linked via Wikipedia’s interlanguage links.

If a wiki doesn’t have a langlink for that article, the scraper won’t magically discover it. (That’s just how Wikipedia organizes equivalences between pages.)




-------------


What’s stored in the DB

Two tables:

categories(lang, title) → records which categories have already been crawled.

pages(lang, title) → records which article pages have already been claimed.

That’s it — no text content, just language + title pairs.

When entries get written

Not up front. The script doesn’t load all articles into the DB first.

Instead, the DB is populated as you go:

When a category is about to be scraped, it calls ledger.claim_category(...).

If it’s new, it’s inserted.

If it’s already there, it’s skipped.

When a batch of pages is about to be fetched, it calls ledger.claim_page(...) or claim_pages_many(...).

If a page hasn’t been seen before, it’s inserted.

If it was already inserted in a previous run or by another worker, it’s skipped.

Why this design

Keeps the DB small (only holds titles you’ve actually encountered).

Supports resumability: if the scraper crashes halfway, rerunning will skip anything already in the DB.

Prevents duplication when you parallelize or run multiple jobs.
