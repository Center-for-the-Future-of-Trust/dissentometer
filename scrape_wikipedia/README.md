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



-----

It’s not multi-process or multi-threaded CPU parallelism. Python’s GIL means CPU-heavy work isn’t sped up; but this scraper is network-bound, so async concurrency is the right tool.




---



---

Update this: 

```
<OUTPUT>/
  History of ideologies/
    en/
      History of ideologies/
        depth_1/
          <root-level-article__hash>/
            English__en.txt
            Français__fr.txt
            ...
          <Subcat A>/                ← subcategory surfaced at depth 1
          <Subcat B>/
        depth_2/
          <Subcat A>/
            <article__hash>/
              English__en.txt
              ...
          <Subcat B>/
            <article__hash>/...
        depth_3/
          <Sub-subcat C>/
            <article__hash>/...
        ...
```

-----


The script’s default is Unicode filenames (ASCII_FILENAMES=False), so without the flag it will keep non-ASCII names.

Notes

This only affects folder/file names, not the article text. The .txt contents are saved in UTF-8 either way.

To confirm you have non-ASCII filenames after a run:
